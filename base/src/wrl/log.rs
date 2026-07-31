use std::cell::Cell;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, sleep, spawn};
use std::time::Duration;

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::base::util::HashMap;

//////////////////////////////////////////////////////////////////////////////

// Off-thread logging helper:

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Handle(usize);

enum Action {
    Open(Handle, Box<str>, Compression),
    Write(Handle, Box<[u8]>),
    Flush(Handle),
    Close(Handle),
    Shutdown,
}

struct LogWorker {
    ops: Arc<Mutex<VecDeque<Action>>>,
    files: HashMap<Handle, Box<dyn Write>>,
}

impl LogWorker {
    fn run(&mut self) {
        while self.process(/*max=*/1024) {
            sleep(Duration::from_millis(1));
        }
    }

    fn process(&mut self, max: usize) -> bool {
        for _ in 0..max {
            let Some(x) = self.ops.lock().unwrap().pop_front() else { break };
            match x {
                Action::Open(handle, path, compression) => self.open(handle, &path, compression),
                Action::Write(handle, data) => self.write(handle, &data),
                Action::Flush(handle) => self.flush(handle),
                Action::Close(handle) => self.close(handle),
                Action::Shutdown => return false,
            }
        }
        true
    }

    fn open(&mut self, handle: Handle, filename: &str, compression: Compression) {
        let file = BufWriter::new(File::create(filename).unwrap());
        let file: Box<dyn Write> = if compression != Compression::none() {
            Box::new(GzEncoder::new(file, compression))
        } else {
            Box::new(file)
        };
        let okay = self.files.insert(handle, file);
        assert!(okay.is_none());
    }

    fn write(&mut self, handle: Handle, data: &[u8]) {
        self.files.get_mut(&handle).unwrap().write_all(data).unwrap()
    }

    fn flush(&mut self, handle: Handle) {
        self.files.get_mut(&handle).unwrap().flush().unwrap()
    }

    fn close(&mut self, handle: Handle) {
        self.files.remove(&handle).unwrap().flush().unwrap()
    }
}

//////////////////////////////////////////////////////////////////////////////

// Logging interface:

pub struct Logger {
    dir: String,
    ops: Arc<Mutex<VecDeque<Action>>>,
    worker: Option<JoinHandle<()>>,
    next_handle: Cell<usize>,
}

impl Drop for Logger {
    fn drop(&mut self) {
        self.ops.lock().unwrap().push_back(Action::Shutdown);
        self.worker.take().unwrap().join().unwrap();
    }
}

impl Logger {
    pub fn new(dir: String) -> Self {
        let ops = Default::default();
        let cloned = Arc::clone(&ops);
        let worker = spawn(move || LogWorker { ops, files: Default::default() }.run());
        Self { dir, ops: cloned, worker: Some(worker), next_handle: Default::default() }
    }

    pub fn open_text(&self, filename: &str) -> LogFile {
        self.open_with(filename, Compression::none())
    }

    pub fn open_compressed(&self, filename: &str) -> LogFile {
        self.open_with(filename, Compression::default())
    }

    fn open_with(&self, filename: &str, compression: Compression) -> LogFile {
        let handle = Handle(self.next_handle.replace(self.next_handle.get() + 1));
        let open = Some((format!("{}/{}", self.dir, filename), compression));
        LogFile { ops: Arc::clone(&self.ops), open, buffer: vec![], handle }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Logging file handle:

pub struct LogFile {
    ops: Arc<Mutex<VecDeque<Action>>>,
    open: Option<(String, Compression)>,
    buffer: Vec<u8>,
    handle: Handle,
}

impl LogFile {
    fn push(&self, op: Action) {
        self.ops.lock().unwrap().push_back(op);
    }
}

impl Drop for LogFile {
    fn drop(&mut self) {
        self.flush().unwrap();
        self.push(Action::Close(self.handle));
    }
}

impl Write for LogFile {
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        self.buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some((path, compression)) = self.open.take() {
            let path = path.into_boxed_str();
            self.push(Action::Open(self.handle, path, compression));
        }
        if !self.buffer.is_empty() {
            let data = std::mem::take(&mut self.buffer).into_boxed_slice();
            self.push(Action::Write(self.handle, data));
        }
        self.push(Action::Flush(self.handle));
        Ok(())
    }
}
