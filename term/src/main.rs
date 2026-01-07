use std::io::{self, Write};

use game_loop::{game_loop, TimeTrait};
use termion::{clear, color};
use termion::cursor::{Goto, Hide, Show};
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::{IntoRawMode, RawTerminal};
use termion::screen::{ToAlternateScreen, ToMainScreen};

use wrl_base::base::{Char, Color, Glyph, Matrix, Point};
use wrl_base::game::{GameMode, Input, State};

type RawMouseTerminal = MouseTerminal<RawTerminal<io::Stdout>>;

struct Screen {
    extent: Point,
    offset: Point,
    output: RawMouseTerminal,
    next: Matrix<Glyph>,
    prev: Matrix<Glyph>,
    fg: Option<Color>,
    bg: Option<Color>,
}

impl Screen {
    fn new(size: Point) -> Self {
        let prev = Matrix::new(size, ' '.into());
        let next = Matrix::new(size, ' '.into());
        let (x, y) = termion::terminal_size().unwrap();
        let output = io::stdout().into_raw_mode().unwrap().into();
        let (fg, bg) = (None, None);
        let extent = Point(x as i32, y as i32);
        let offset = Point((extent - size).0 / 2 + 1, (extent - size).1 / 2 + 1);
        Self { extent, offset, output, next, prev, fg, bg }
    }

    fn render(&mut self, stats: &Stats, delta: f64) -> io::Result<()> {
        let mut lines_changed = 0;
        let Point(sx, sy) = self.next.size;
        for y in 0..sy {
            let mut start = sx;
            let mut limit = 0;
            for x in 0..sx {
                let next = self.next.get(Point(x, y));
                let prev = self.prev.get(Point(x, y));
                if  next == prev { continue; }
                start = std::cmp::min(start, x);
                limit = std::cmp::max(limit, x);
            }
            if start > limit { continue; }

            lines_changed += 1;
            let mx = (self.offset.0 + start) as u16;
            let my = (self.offset.1 + y) as u16;
            write!(self.output, "{}", Goto(mx, my))?;

            let mut x = start;
            while x <= limit {
                let glyph = self.next.get(Point(x, y));
                self.set_foreground(glyph.fg())?;
                self.set_background(glyph.bg())?;
                x += self.write_char(glyph.ch())?;
            }
        }
        std::mem::swap(&mut self.next, &mut self.prev);

        if delta > 1.0 {
            let out = format!(
                   "Update: {:.2}% / Render: {:.2}% / FPS: {:.2}",
                   100.0 * stats.update_total.get() / delta,
                   100.0 * stats.render_total.get() / delta,
                   stats.ticks.get() as f64 / delta);
            self.write_status_message(&out)?;
            lines_changed += 1;
            stats.clear();
        }

        if lines_changed > 0 { self.output.flush() } else { Ok(()) }
    }

    fn enter_alt_screen(&mut self) -> io::Result<()> {
        write!(self.output, "{}{}{}", ToAlternateScreen, Hide, clear::All)?;
        self.reset_colors()?;
        self.output.flush()
    }

    fn exit_alt_screen(&mut self) -> io::Result<()> {
        self.reset_colors()?;
        write!(self.output, "{}{}", ToMainScreen, Show)?;
        self.output.flush()
    }

    fn reset_colors(&mut self) -> io::Result<()> {
        self.clear_foreground()?;
        self.clear_background()
    }

    fn clear_foreground(&mut self) -> io::Result<()> {
        self.fg = None;
        write!(self.output, "{}", color::Fg(color::Reset))
    }

    fn clear_background(&mut self) -> io::Result<()> {
        self.bg = None;
        write!(self.output, "{}", color::Bg(color::Reset))
    }

    fn set_foreground(&mut self, color: Color) -> io::Result<()> {
        if self.fg == Some(color) { return Ok(()); }
        self.fg = Some(color);
        let r = ((color.0 >> 16) & 0xff) as u8;
        let g = ((color.0 >> 8) & 0xff) as u8;
        let b = (color.0 & 0xff) as u8;
        write!(self.output, "{}", color::Fg(color::Rgb(r, g, b)))
    }

    fn set_background(&mut self, color: Color) -> io::Result<()> {
        if self.bg == Some(color) { return Ok(()); }
        self.bg = Some(color);
        let r = ((color.0 >> 16) & 0xff) as u8;
        let g = ((color.0 >> 8) & 0xff) as u8;
        let b = (color.0 & 0xff) as u8;
        write!(self.output, "{}", color::Bg(color::Rgb(r, g, b)))
    }

    fn write_char(&mut self, ch: Char) -> io::Result<i32> {
        if ch.0 == 0xff00 {
            write!(self.output, "  ")?;
            Ok(2)
        } else {
            write!(self.output, "{}", char::from_u32(ch.0 as u32).unwrap())?;
            Ok(if ch.is_wide() { 2 } else { 1 })
        }
    }

    fn write_status_message(&mut self, msg: &str) -> io::Result<()> {
        self.clear_foreground()?;
        self.clear_background()?;
        let x = (self.extent.0 - msg.len() as i32) as u16;
        let y = self.extent.1 as u16;
        write!(self.output, "{}{}{}", Goto(x, y), clear::CurrentLine, msg)
    }
}

#[derive(Default)]
struct Stats {
    render_total: std::cell::Cell<f64>,
    update_total: std::cell::Cell<f64>,
    ticks: std::cell::Cell<usize>,
}

impl Stats {
    fn clear(&self) {
        self.render_total.set(0.0);
        self.update_total.set(0.0);
        self.ticks.set(0);
    }

    fn on_render(&self, delta: f64) {
        self.render_total.set(self.render_total.get() + delta);
    }

    fn on_update(&self, delta: f64) {
        self.update_total.set(self.update_total.get() + delta);
    }
}

fn key_input(key: Key) -> Option<Input> {
    match key {
        Key::BackTab => Some(Input::BackTab),
        Key::Char(ch) => Some(Input::Char(ch)),
        Key::Esc => Some(Input::Escape),
        _ => None,
    }
}

fn mouse_input(mouse: MouseEvent, screen: &Screen) -> Option<Input> {
    match mouse {
        MouseEvent::Press(MouseButton::Left, x, y) => {
            let pos = Point(x as i32, y as i32) - screen.offset;
            if screen.prev.contains(pos) { Some(Input::Click(pos)) } else { None }
        },
        _ => None,
    }
}

fn input(event: Event, screen: &Screen) -> Option<Input> {
    match event {
        Event::Key(key) => key_input(key),
        Event::Mouse(mouse) => mouse_input(mouse, screen),
        _ => None,
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let count = args.len();

    let mode = if count < 2 { GameMode::Play } else {
        match args[1].as_str() {
            "--debug" if count == 2 => GameMode::Debug,
            "--gym"   if count == 3 => GameMode::Gym,
            "--sim"   if count == 3 => GameMode::Sim,
            _ => panic!("Usage: wrl-term (--debug|--gym $COUNT|--sim $COUNT)?"),
        }
    };
    let game = State::new(/*seed=*/None, mode);

    if matches!(mode, GameMode::Gym | GameMode::Sim) {
        let mut game = game;
        let turns = args[2].parse::<usize>().unwrap();
        let gym = mode == GameMode::Gym;
        for i in 0..turns {
            if !gym { println!("Iteration: {}", i); }
            game.add_input(Input::Char('.'));
            game.update();
        }
        return;
    }

    let mut output = Matrix::default();
    game.render(&mut output);

    let mut events = termion::async_stdin().events();
    let mut screen = Screen::new(output.size);
    screen.enter_alt_screen().unwrap();
    screen.write_status_message("<calculating FPS...>").unwrap();
    screen.output.flush().unwrap();

    let stats = Stats::default();
    let mut time = game_loop::Time::now();

    game_loop(game, 60, 0.01, |g| {
        let start = game_loop::Time::now();
        if let Some(Ok(e)) = events.next() {
            match e {
                Event::Key(Key::Ctrl('c')) => g.exit(),
                x => if let Some(x) = input(x, &screen) { g.game.add_input(x); }
            }
        }
        g.game.update();
        let limit = game_loop::Time::now();
        stats.on_update(limit.sub(&start));

        // It's a roguelike: without an update(), render() will not change,
        // and render() is a fast operation. We frame-lock renders to updates.
        let start = game_loop::Time::now();
        let delta = start.sub(&time);
        g.game.render(&mut screen.next);
        screen.render(&stats, delta).unwrap();
        let limit = game_loop::Time::now();
        stats.on_render(limit.sub(&start));

        // Reset stats after they're printed as a status message.
        let ticks = stats.ticks.get();
        stats.ticks.set(ticks + 1);
        if ticks == 0 {
            time = start;
        }
    }, |_| {
        std::thread::sleep(std::time::Duration::from_micros(1000));
    });

    screen.exit_alt_screen().unwrap();
    screen.output.suspend_raw_mode().unwrap();
}
