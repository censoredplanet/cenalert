import * as Plot from 'npm:@observablehq/plot';
import * as d3 from 'npm:d3';

export function createSearchVolumeChart(
  timeseries,
  { zoomedAnomalies = [], showHighlight = true, width = 800, onPlotClick } = {},
) {
  // Formatter helpers
  const fmtYMD = d3.utcFormat('%Y.%m.%d');
  const fmtDMY = d3.utcFormat('%d.%m.%Y');

  const plotColors = {
    text: 'var(--plot-text)',
    line: 'var(--plot-line)',
    grid: 'var(--plot-grid)',
    bg: 'var(--plot-bg)',
  };
  const y2 = d3.max(timeseries, (d) => d.rate);

  const marks = [
    Plot.ruleY([0], { stroke: plotColors.grid }),
    Plot.lineY(timeseries, {
      x: 'date',
      y: 'rate',
      stroke: plotColors.line,
      tip: true,
      title: (d) =>
        `Topic: ${d.topic || 'Unknown'}\nDate: ${fmtYMD(d.date)}\nRate: ${d.rate?.toFixed(2) ?? 'N/A'}`,
    }),
  ];

  if (showHighlight && zoomedAnomalies.length) {
    marks.push(
      Plot.rectY(zoomedAnomalies, {
        x1: (d) => d.s,
        x2: (d) => d.e,
        y1: y2,
        y2: 0,
        fill: '#df9d81',
        fillOpacity: 0.35,
        stroke: '#f56363',
        strokeOpacity: 0.6,
        strokeWidth: 0.7,
        tip: true,
        title: (d) =>
          `Cause: ${d.cause}\nDuration: ${fmtDMY(d.s)} – ${fmtDMY(d.e)}\n${d.impact ? `Impact: ${(+d.impact).toFixed(2)}` : ''}`,
      }),
      Plot.ruleX(
        zoomedAnomalies.map((d) => d.s),
        { stroke: '#ef4444', strokeOpacity: 0.6, strokeWidth: 0.7 },
      ),
      Plot.ruleX(
        zoomedAnomalies.map((d) => d.e),
        { stroke: '#ef4444', strokeOpacity: 0.6, strokeWidth: 0.7 },
      ),
    );
  }

  const plotSvg = Plot.plot({
    style: {
      background: plotColors.bg,
      color: plotColors.text,
      fontSize: '13px',
    },
    width,
    grid: true,
    y: { grid: true, label: '', stroke: plotColors.grid },
    x: { label: '', stroke: plotColors.grid },
    marks,
  });

  // Attach generic click listener
  if (onPlotClick) {
    plotSvg.addEventListener('click', () => {
      if (plotSvg.value?.date) {
        onPlotClick(plotSvg.value.date);
      }
    });
  }

  return plotSvg;
}
