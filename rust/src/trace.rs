//! Debug tracing: stderr on native (AUTOFOCUS_DEBUG=1), an in-memory
//! buffer on WASM that consumers read back via `last_trace()`.

#[cfg(target_arch = "wasm32")]
thread_local! {
    pub static TRACE: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}

#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! dbg_focus {
    ($($arg:tt)*) => {
        if std::env::var_os("AUTOFOCUS_DEBUG").is_some() { eprintln!($($arg)*); }
    };
}
// On WASM the stage trace is captured into a buffer consumers read via
// last_trace() — the browser console is the debugger there, not stderr.
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! dbg_focus {
    ($($arg:tt)*) => {
        $crate::trace::TRACE.with(|t| {
            use std::fmt::Write;
            let _ = writeln!(t.borrow_mut(), $($arg)*);
        })
    };
}
