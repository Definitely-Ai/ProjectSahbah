/**
 * Logger — structured logging with UI panel and console output.
 *
 * Provides four levels: debug, info, warn, error.
 * All messages go to both the developer console and the on-screen log panel
 * so that end users can report issues with full context.
 */

const Logger = (() => {
  const MAX_LOG_ENTRIES = 200;
  const entries = [];

  const levelColors = {
    debug: '#64748b',
    info:  '#14b8a6',
    warn:  '#f59e0b',
    error: '#ef4444',
  };

  const levelClass = {
    debug: 'log-debug',
    info:  'log-info',
    warn:  'log-warn',
    error: 'log-error',
  };

  function timestamp() {
    const d = new Date();
    return d.toLocaleTimeString('en-US', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
  }

  function log(level, ...args) {
    const ts = timestamp();
    const msg = args.map(a =>
      typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a)
    ).join(' ');

    // Console output with color
    const consoleFn = level === 'error' ? console.error
      : level === 'warn' ? console.warn
      : level === 'debug' ? console.debug
      : console.log;
    consoleFn(`%c[${level.toUpperCase()}] ${ts}%c ${msg}`, `color:${levelColors[level]};font-weight:bold`, 'color:inherit');

    // Store entry
    const entry = { level, ts, msg };
    entries.push(entry);
    if (entries.length > MAX_LOG_ENTRIES) entries.shift();

    // Update UI panel
    _renderEntry(entry);
  }

  function _renderEntry(entry) {
    const panel = document.getElementById('logContent');
    if (!panel) return;

    const div = document.createElement('div');
    div.className = `log-entry ${levelClass[entry.level]}`;
    div.textContent = `[${entry.ts}] ${entry.level.toUpperCase()}: ${entry.msg}`;
    panel.appendChild(div);

    // Auto-scroll
    panel.scrollTop = panel.scrollHeight;
  }

  function getEntries() {
    return [...entries];
  }

  function clear() {
    entries.length = 0;
    const panel = document.getElementById('logContent');
    if (panel) panel.innerHTML = '';
  }

  function exportLog() {
    return entries.map(e => `[${e.ts}] ${e.level.toUpperCase()}: ${e.msg}`).join('\n');
  }

  return {
    debug: (...args) => log('debug', ...args),
    info:  (...args) => log('info', ...args),
    warn:  (...args) => log('warn', ...args),
    error: (...args) => log('error', ...args),
    getEntries,
    clear,
    exportLog,
  };
})();

// Make Logger available globally
window.Logger = Logger;
