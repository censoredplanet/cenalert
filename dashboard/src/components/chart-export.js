const SVG_NS = 'http://www.w3.org/2000/svg';

const SCALE = 2;
const PAD = 24;
const LOGO_HEIGHT = 30;
const FONT =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

const LIGHT_THEME = {
  '--theme-foreground': '#1b1e23',
  '--theme-background': '#ffffff',
  '--theme-background-alt': '#f5f7fa',
  '--theme-foreground-muted': '#6e7684',
  '--theme-foreground-faint': '#c4c9d0',
  '--theme-foreground-faintest': '#e6e9ee',
  '--color-text-primary': '#1a1a1a',
  '--color-text-secondary': '#444444',
  '--text': '#222222',
  '--text-light': '#555555',
  '--plot-text': '#222222',
  '--plot-line': '#222222',
  '--plot-grid': '#555555',
  '--plot-bg': 'transparent',
};

const STYLE_PROPS = [
  'fill',
  'fill-opacity',
  'stroke',
  'stroke-width',
  'stroke-opacity',
  'stroke-dasharray',
  'stroke-linecap',
  'stroke-linejoin',
  'opacity',
  'visibility',
  'display',
  'font-family',
  'font-size',
  'font-weight',
  'font-style',
  'letter-spacing',
  'text-anchor',
  'dominant-baseline',
  'paint-order',
];

function freezeStyles(clone) {
  const stage = document.createElement('div');
  stage.setAttribute(
    'style',
    'position:absolute;left:-99999px;top:0;width:2000px;' +
      `color:${LIGHT_THEME['--theme-foreground']};font-family:${FONT};`,
  );
  for (const [name, value] of Object.entries(LIGHT_THEME)) {
    stage.style.setProperty(name, value);
  }
  document.body.append(stage);
  stage.append(clone);

  try {
    const nodes = [clone, ...clone.querySelectorAll('*')];
    const frozen = nodes.map((node) => {
      const computed = window.getComputedStyle(node);
      let css = '';
      for (const prop of STYLE_PROPS) {
        const value = computed.getPropertyValue(prop);
        if (value) css += `${prop}:${value};`;
      }
      return css;
    });
    nodes.forEach((node, i) => node.setAttribute('style', frozen[i]));
  } finally {
    stage.remove();
  }
}

async function inlineAsset(url) {
  const response = await fetch(url);
  const markup = await response.text();
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
}

const NOTE_SIZE = 11;
const NOTE_LEADING = 15;
const CAPTION_SIZE = 11;
const CAPTION_LEADING = 14;
const CREDIT = 'dashboard.censoredplanet.org';

function measureText(text, fontSize, weight = 400) {
  const context = document.createElement('canvas').getContext('2d');
  context.font = `${weight} ${fontSize}px ${FONT}`;
  return context.measureText(text).width;
}

function wrapText(text, maxWidth, fontSize, firstLineWidth = maxWidth) {
  const context = document.createElement('canvas').getContext('2d');
  context.font = `${fontSize}px ${FONT}`;

  const lines = [];
  let line = '';
  for (const word of String(text).split(/\s+/).filter(Boolean)) {
    const candidate = line ? `${line} ${word}` : word;
    const limit = lines.length === 0 ? firstLineWidth : maxWidth;
    if (line && context.measureText(candidate).width > limit) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function svgEl(name, attrs) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

function textEl(content, attrs) {
  const el = svgEl('text', { 'font-family': FONT, ...attrs });
  el.textContent = content;
  return el;
}

/**
 * Renders a chart as a PNG poster and triggers a download.
 *
 * The chart is cloned from the live DOM rather than re-rendered, so whatever is
 * on screen is what gets exported — including optional layers the reader has
 * toggled off. Describe that state in `meta` so the image is self-explanatory
 * once it leaves the dashboard.
 *
 * @param {object} options
 * @param {SVGElement} options.chart live chart node to capture
 * @param {string} options.title heading printed above the chart
 * @param {Array<[string, string]>} options.meta label/value pairs for the caption
 * @param {string} options.logoUrl URL of an SVG logo, e.g. FileAttachment(…).href
 * @param {string} [options.notes] free text rendered below the chart, wrapped
 * @param {string} [options.notesLabel] bold prefix for the notes block
 * @param {string} options.filename download name, without extension
 * @param {string} [options.background] poster background colour
 */
export async function exportChartPng({
  chart,
  title,
  meta,
  notes,
  notesLabel,
  logoUrl,
  filename,
  background = '#ffffff',
}) {
  const box = chart.getBoundingClientRect();
  const chartWidth = Math.round(box.width);
  const bbox = chart.getBBox();
  const chartHeight = Math.ceil(Math.max(box.height, bbox.y + bbox.height)) + 4;
  const width = chartWidth + PAD * 2;
  const captionTop = PAD + LOGO_HEIGHT + 42;
  const creditWidth = measureText(CREDIT, CAPTION_SIZE);
  const creditFitsBesideLogo =
    width - PAD - creditWidth > PAD + LOGO_HEIGHT * 6 + 12;
  const captionText = [
    ...meta.map(([label, value]) => `${label}: ${value}`),
    ...(creditFitsBesideLogo ? [] : [CREDIT]),
  ].join('   ·   ');
  const captionLines = wrapText(captionText, chartWidth, CAPTION_SIZE);
  const headerHeight =
    captionTop + (captionLines.length - 1) * CAPTION_LEADING + 10;
  const notePrefix = notes && notesLabel ? `${notesLabel}: ` : '';
  const prefixWidth = notePrefix ? measureText(notePrefix, NOTE_SIZE, 600) : 0;
  const noteLines = notes
    ? wrapText(notes, chartWidth, NOTE_SIZE, chartWidth - prefixWidth)
    : [];
  const notesHeight = noteLines.length
    ? noteLines.length * NOTE_LEADING + PAD
    : 0;
  const height = headerHeight + chartHeight + notesHeight + PAD;

  const clone = chart.cloneNode(true);
  clone.setAttribute('width', chartWidth);
  clone.setAttribute('height', chartHeight);
  freezeStyles(clone);

  const poster = svgEl('svg', {
    xmlns: SVG_NS,
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
  });

  poster.append(svgEl('rect', { width, height, fill: background }));

  poster.append(
    svgEl('image', {
      href: await inlineAsset(logoUrl),
      x: PAD,
      y: PAD,
      height: LOGO_HEIGHT,
      width: LOGO_HEIGHT * 6,
      preserveAspectRatio: 'xMinYMid meet',
    }),
  );

  poster.append(
    textEl(title, {
      x: PAD,
      y: PAD + LOGO_HEIGHT + 24,
      'font-size': 16,
      'font-weight': 600,
      fill: '#1a1a1a',
    }),
  );

  const caption = textEl('', {
    x: PAD,
    y: captionTop,
    'font-size': CAPTION_SIZE,
    fill: '#555555',
  });
  captionLines.forEach((line, i) => {
    const span = document.createElementNS(SVG_NS, 'tspan');
    span.setAttribute('x', PAD);
    if (i > 0) span.setAttribute('dy', CAPTION_LEADING);
    span.textContent = line;
    caption.append(span);
  });
  poster.append(caption);

  const group = svgEl('g', { transform: `translate(${PAD}, ${headerHeight})` });
  group.append(clone);
  poster.append(group);

  if (noteLines.length) {
    const note = textEl('', {
      x: PAD,
      y: headerHeight + chartHeight + PAD,
      'font-size': NOTE_SIZE,
      fill: '#444444',
    });
    noteLines.forEach((line, i) => {
      if (i === 0 && notePrefix) {
        const label = document.createElementNS(SVG_NS, 'tspan');
        label.setAttribute('x', PAD);
        label.setAttribute('font-weight', '600');
        label.setAttribute('fill', '#1a1a1a');
        label.textContent = notePrefix;
        note.append(label);

        const rest = document.createElementNS(SVG_NS, 'tspan');
        rest.textContent = line;
        note.append(rest);
        return;
      }

      const span = document.createElementNS(SVG_NS, 'tspan');
      span.setAttribute('x', PAD);
      if (i > 0) span.setAttribute('dy', NOTE_LEADING);
      span.textContent = line;
      note.append(span);
    });
    poster.append(note);
  }

  if (creditFitsBesideLogo) {
    poster.append(
      textEl(CREDIT, {
        x: width - PAD,
        y: PAD + 20,
        'font-size': CAPTION_SIZE,
        fill: '#888888',
        'text-anchor': 'end',
      }),
    );
  }

  const markup = new XMLSerializer().serializeToString(poster);
  const image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error('Could not rasterise the chart'));
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
  });

  const canvas = document.createElement('canvas');
  canvas.width = width * SCALE;
  canvas.height = height * SCALE;
  const context = canvas.getContext('2d');
  context.scale(SCALE, SCALE);
  context.drawImage(image, 0, 0);

  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, 'image/png'),
  );
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = `${filename}.png`;
  link.click();
  URL.revokeObjectURL(href);
}
