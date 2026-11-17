import * as d3 from 'npm:d3';
import * as htl from 'npm:htl';

export const formatDMYdots = (s) => {
  if (!s) return '';

  const y = s.slice(0, 4);
  const m = s.slice(5, 7);
  const d = s.slice(8, 10);

  return y && m && d ? `${d}.${m}.${y}` : '';
};

export function updateURL(paramsObj = {}, clearEvent = false) {
  const currentParams = new URLSearchParams(window.location.search);

  for (const [key, val] of Object.entries(paramsObj)) {
    if (val === null || val === undefined || val === '') {
      currentParams.delete(key);
    } else {
      currentParams.set(key, val);
    }
  }

  if (clearEvent) currentParams.delete('event');

  const query = currentParams.toString();
  const newURL = query
    ? `${window.location.pathname}?${query}`
    : window.location.pathname;

  window.history.pushState({}, '', newURL);
}

export function fmt(d) {
  return d.toISOString().slice(0, 10);
}

export function leafColor(name) {
  if (name.startsWith('✅')) return '#0bba13';
  if (name.startsWith('❗️')) return '#9e0202';
  if (name.startsWith('❓')) return '#f56565';
  if (name.startsWith('❔')) return '#ada6a6cb';
  return '#bbbbbb';
}

const LIGHTNESS_SPAN = 46;

/**
 * Builds a colour lookup that gives every outcome its own shade.
 *
 * @param {Map<string, number>|Iterable<[string, number]>} counts totals per
 *     outcome label, as produced by the chart's own data
 * @return {(name: string) => string}
 */
export function makeLeafColor(counts) {
  const totals = counts instanceof Map ? counts : new Map(counts);

  const families = new Map();
  for (const name of totals.keys()) {
    const key = ['✅', '❗️', '❓', '❔'].find((s) => name.startsWith(s)) ?? '';
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(name);
  }
  for (const members of families.values()) {
    members.sort(
      (a, b) =>
        (totals.get(b) ?? 0) - (totals.get(a) ?? 0) || a.localeCompare(b),
    );
  }

  const colors = new Map();
  for (const [key, members] of families) {
    const base = d3.hcl(leafColor(key || 'other'));
    members.forEach((name, i) => {
      const offset =
        members.length > 1 ? (i / (members.length - 1)) * LIGHTNESS_SPAN : 0;
      const shade = d3.hcl(base);
      shade.l = Math.min(80, base.l + offset);
      colors.set(name, shade.formatHex());
    });
  }

  return (name) => colors.get(name) ?? leafColor(name);
}

export function sparkbar() {
  return (x) =>
    htl.html`<div class="sparkbar" style="width: ${x}%;">${x.toLocaleString('en-US')}%</div>`;
}

export function updateBounds(start, end, dataMaxDate, dataMinDate) {
  const s = new Date(start.value);
  const e = new Date(end.value);
  const maxAllowedEnd = new Date(s);
  maxAllowedEnd.setMonth(maxAllowedEnd.getMonth() + 6);
  if (maxAllowedEnd > dataMaxDate) maxAllowedEnd.setTime(dataMaxDate.getTime());

  end.setAttribute('min', fmt(s));
  end.setAttribute('max', fmt(maxAllowedEnd));
  if (e < s) end.value = fmt(s);
  if (e > maxAllowedEnd) end.value = fmt(maxAllowedEnd);

  const minAllowedStart = new Date(e);
  minAllowedStart.setMonth(minAllowedStart.getMonth() - 6);
  if (minAllowedStart < dataMinDate)
    minAllowedStart.setTime(dataMinDate.getTime());

  start.setAttribute('min', fmt(minAllowedStart));
  start.setAttribute('max', fmt(e));
  if (s < minAllowedStart) start.value = fmt(minAllowedStart);
  if (s > e) start.value = fmt(e);
}

export function initTableTooltip() {
  if (typeof document === 'undefined' || window._tableTooltipBound) return;

  let _tip = document.getElementById('table-tooltip');
  if (!_tip) {
    _tip = document.createElement('div');
    _tip.id = 'table-tooltip';
    document.body.appendChild(_tip);
  }

  window._tableTooltipBound = true;

  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest('.cell-ellipsis');
    if (!el) return;
    const text = el.dataset.full || el.textContent || '';
    if (!text.trim()) return;
    _tip.textContent = text;
    _tip.style.opacity = '1';
  });

  document.addEventListener('mousemove', (e) => {
    if (_tip.style.opacity !== '1') return;
    const pad = 16;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const x = Math.min(e.clientX + 12, vw - pad);
    const y = Math.min(e.clientY + 18, vh - pad);
    _tip.style.left = `${x}px`;
    _tip.style.top = `${y}px`;
  });

  document.addEventListener('mouseout', (e) => {
    if (e.target.closest('.cell-ellipsis')) {
      _tip.style.opacity = '0';
    }
  });
}
