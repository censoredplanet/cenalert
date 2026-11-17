import { downloadLinks } from './data-download.js';
import { exportChartPng } from './chart-export.js';

export function createDetailOpener(deps) {
  const {
    html,
    d3,
    Plot,
    resize,
    DAY,
    PX_PADDING,
    events,
    formatImpact,
    detailSection,
    startDate,
    endDate,
    logoUrl,
  } = deps;

  function updateEventUrl(selectedEvent) {
    const params = new URLSearchParams(window.location.search);

    if (!selectedEvent) {
      if (params.has('event')) {
        params.delete('event');
        const newUrl = params.toString()
          ? `${window.location.pathname}?${params.toString()}`
          : window.location.pathname;
        window.history.replaceState({}, '', newUrl);
      }
      return;
    }

    params.set(
      'event',
      selectedEvent.__matchKey || selectedEvent.dateLabel || '',
    );
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }

  function renderDetail(code, name, countryEvents, countryHasAnyEvents) {
    function twemojiFlagCode(code) {
      return [...code.toUpperCase()]
        .map((c) => (c.codePointAt(0) - 0x41 + 0x1f1e6).toString(16))
        .join('-');
    }

    const flag = html`
      <img
        src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/${twemojiFlagCode(
          code,
        )}.svg"
      />
    `;

    const rangeStart = d3.min(countryEvents, (d) => d.date);
    const rangeEnd = d3.max(countryEvents, (d) => d.date);
    detailSection.innerHTML = '';
    const fmtYMDdots = d3.utcFormat('%Y.%m.%d');
    const dateRangeLabel =
      rangeStart && rangeEnd
        ? `${fmtYMDdots(rangeStart)} - ${fmtYMDdots(rangeEnd)}`
        : '';
    const heading = html`<h2 class="detail-title">
      <span class="flag">${flag}</span>
      <span class="country-name">Events in ${name}</span>
      ${dateRangeLabel
        ? html`<span class="date-range">(${dateRangeLabel})</span>`
        : ''}
    </h2>`;
    const legend = html`
      <div class="top-legend-wrap">
        <div class="legend-desc">
          <strong>Impact Score:</strong> Scores the severity of an event based
          on how high and how long VPN searches surged, helping prioritize
          investigations.
        </div>
        <div class="legend-scale-container">
          <span class="legend-label">Low</span>
          <div class="legend-circles">
            <div class="circle c-low" title="Low Impact"></div>
            <div class="circle c-mod" title="Moderate Impact"></div>
            <div class="circle c-high" title="High Impact"></div>
            <div class="circle c-sev" title="Severe Impact"></div>
          </div>
          <span class="legend-label">Severe</span>
        </div>
      </div>
    `;

    const detailHeader = html`<div class="detail-header-container">
      ${heading}${legend}
    </div>`;

    detailSection.append(detailHeader);
    if (!countryEvents.length) {
      if (!countryHasAnyEvents) {
        detailSection.append(html`<div class="empty">No events.</div>`);
      } else {
        detailSection.append(html`
          <div class="empty">
            No events within the selected time range. Try expanding the date
            range or changing filters.
          </div>
        `);
      }
      return;
    }
    const container = html`<div class="detail-grid">
      <div class="left-col">
        <div class="events-scroller"></div>
      </div>
      <div class="center-col">
        <div class="graph-wrap"></div>
      </div>
      <div class="right-col">
        <div class="summary-wrap"></div>
      </div>
    </div>`;

    detailSection.append(container);

    const scroller = container.querySelector('.events-scroller');
    const graphWrap = container.querySelector('.graph-wrap');
    const summaryWrap = container.querySelector('.summary-wrap');

    // Graph (center column)
    const rightTitle = html`<h3 class="right-title">Rate over time</h3>`;
    const rightBody = html`<div class="right-body"></div>`;
    graphWrap.append(rightTitle, rightBody);

    // Summary (right column)
    const summaryTitle = html`<h3 class="summary-title">Event Summary</h3>`;
    const summaryBody = html`<div id="summary-body" class="summary-body">
      Select an event to see details.
    </div>`;

    const cleanEvents = countryEvents.map((d) => ({
      country: name,
      startDate: d.startDate,
      endDate: d.endDate,
      who: d.who,
      description: d.description,
      impact: d.title,
    }));

    const footer = downloadLinks(
      cleanEvents,
      'cp-cenalert',
      'events',
      name,
      rangeStart,
      rangeEnd,
    );

    summaryWrap.append(summaryTitle, summaryBody, footer);

    // initial sizes
    let {
      widthNow: currentWidth,
      r,
      baseMinRow,
      nodeGap,
      topOffset,
    } = computeSizes();
    const margin = { top: 24, right: 10, bottom: 24, left: 20 };

    let width = graphWrap.clientWidth;
    let laneX = margin.left + r;

    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    const VISIBLE_ROWS = isMobile ? 2 : 5;

    function computeSizes() {
      const widthNow = Math.max(280, graphWrap.clientWidth || width || 480);
      const compact = widthNow < 640;
      const computedR = Math.round(
        compact
          ? Math.max(24, Math.min(44, widthNow * 0.11))
          : Math.max(40, Math.min(140, widthNow * 0.15)),
      );
      const computedBaseMinRow = compact
        ? Math.max(50, Math.round(1.6 * computedR))
        : Math.max(80, Math.round(0.9 * computedR + 40));
      const computedNodeGap = compact
        ? Math.max(14, Math.round(computedR * 0.35))
        : Math.max(24, Math.round(computedR * 0.8));
      const computedTopOffset = compact ? computedR + 8 : computedR * 0.6 + 40;
      return {
        widthNow,
        r: computedR,
        baseMinRow: computedBaseMinRow,
        nodeGap: computedNodeGap,
        topOffset: computedTopOffset,
      };
    }

    let height = graphWrap.clientHeight || 600;

    const svg = d3
      .create('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMinYMin meet')
      .style('height', `${height}px`);

    const g = svg.append('g');

    const node = g
      .selectAll('.node')
      .data(countryEvents.map((d, i) => ({ ...d, i })))
      .join('g')
      .attr('class', 'node')
      .attr('transform', () => `translate(${laneX}, ${margin.top + r})`);
    node
      .append('rect')
      .attr('class', 'event-tile')
      .attr('rx', 8) // Rounded corners
      .attr('ry', 8)
      .attr('fill', 'transparent');

    const impactColors = ['#f7f7f7', '#fddbc7', '#f4a582', '#d6604d'];
    function getContrastColor(color) {
      const rgb = d3.color(color);
      if (!rgb) return '#000';
      const luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
      return luminance < 140 ? '#fff' : '#000';
    }

    node
      .append('circle')
      .attr('r', r)
      .attr('fill', (d) => impactColors[d.impactQuartile ?? 0])
      .attr('stroke', '#444')
      .attr('stroke-width', 1.5);

    node
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.25em')
      .style('font-size', `${Math.round(r * 0.35)}px`)
      .text((d) => d.code)
      .attr('fill', (d) =>
        getContrastColor(impactColors[d.impactQuartile ?? 0]),
      );

    // Impact Score
    node
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.2em')
      .style('font-size', `${Math.round(r * 0.35)}px`)
      .attr('font-weight', 700)
      .text((d) => d.title)
      .attr('fill', (d) =>
        getContrastColor(impactColors[d.impactQuartile ?? 0]),
      );

    node
      .on('mouseenter', function () {
        const tile = d3.select(this).select('.event-tile');
        tile
          .transition()
          .duration(120)
          .attr('stroke', 'rgba(30,144,255,0.4)')
          .attr('fill', 'rgba(30,144,255,0.05)');
      })
      .on('mouseleave', function () {
        const self = d3.select(this);
        const tile = self.select('.event-tile');
        if (!self.classed('active')) {
          tile
            .transition()
            .duration(120)
            .attr('stroke', 'transparent')
            .attr('fill', 'transparent');
        }
      })
      .on('click', function (event, d) {
        const self = d3.select(this);
        const isActive = self.classed('active');
        g.selectAll('.node')
          .classed('active', false)
          .select('circle')
          .attr('stroke', '#444')
          .attr('stroke-width', 1.5);
        if (!isActive) {
          self
            .classed('active', true)
            .select('circle')
            .attr('stroke', '#17827B')
            .attr('stroke-width', 5);
          self
            .select('.event-tile')
            .attr('stroke', '#17827B')
            .attr('fill', '#17827B');
        }
        renderRight(d);
        const summaryBody = document.getElementById('summary-body');
        if (summaryBody) {
          const impactLabels = ['Low', 'Moderate', 'High', 'Severe'];
          const impactLevel = impactLabels[d.impactQuartile ?? 0];
          const impactScore = d.title || '—';
          summaryBody.replaceChildren();
          const fields = [
            ['Date', d.dateLabel],
            ['Impact score', impactScore],
            ['Level', impactLevel],
            ['Title', d.eventTitle],
            ['Context', d.description],
            ['Category', d.category],
            ['Cause', d.cause],
            ['Government justification', d.justification],
            ['Affected services', d.affectedServices?.join(', ')],
          ];
          for (const [label, value] of fields) {
            if (value)
              summaryBody.append(
                html`<div><strong>${label}: </strong>${value}</div>`,
              );
          }
          for (const source of d.sources ?? []) {
            const { name, url } = source;
            const label = name || url;
            if (!label) continue;
            if (typeof url === 'string' && /^https?:\/\/\S+$/i.test(url)) {
              summaryBody.append(
                html`<div>
                  <a href=${url} target="_blank" rel="noopener noreferrer"
                    >${label}</a
                  >
                </div>`,
              );
            } else {
              summaryBody.append(html`<div>${label}</div>`);
            }
          }
        }
        updateEventUrl(d);
      });

    let labelDx = r + 12;
    let labelWidth = 0;

    function computeLabelWidth() {
      const el = svg.node();
      const rectW = el.getBoundingClientRect().width;
      const svgPx = rectW || el.clientWidth || width;
      labelWidth = Math.max(380, svgPx - (laneX + labelDx) - margin.right);
      g.selectAll('foreignObject').attr('width', labelWidth);
      g.selectAll('.event-tile').attr('width', labelDx + labelWidth + 24);
    }

    const label = node
      .append('g')
      .attr('transform', `translate(${labelDx}, 0)`);
    label
      .append('line')
      .attr('x1', -8)
      .attr('x2', 0)
      .attr('y1', 0)
      .attr('y2', 0)
      .attr('stroke', '#333');
    label
      .append('text')
      .attr('font-weight', 500)
      .attr('y', r * 0.1)
      .attr('fill', '#333')
      .text((d) => `${d.dateLabel}`)
      .style('font-size', `${Math.round(r * 0.35)}px`);

    const fo = node
      .append('foreignObject')
      .attr('x', labelDx)
      .attr('y', r * 0.1)
      .attr('width', (d) => d.tileWidth)
      .attr('height', 10);

    function renderRight(selectedEvent = null) {
      const fmtYMD = d3.utcFormat('%Y.%m.%d');
      const titleEl = graphWrap.querySelector('.right-title');
      const bodyEl = graphWrap.querySelector('.right-body');
      const mobileExtra = window.matchMedia('(max-width: 768px)').matches
        ? 120
        : 0;

      titleEl.textContent = selectedEvent
        ? selectedEvent.dateLabel
        : 'Rate over time';
      bodyEl.innerHTML = '';

      const seriesnew = deps.seriesForSelectedCountry;

      if (!seriesnew?.length) {
        bodyEl.append(html`<div class="empty">No time series.</div>`);
        return;
      }
      let seriesFiltered = seriesnew;
      if (startDate || endDate) {
        seriesFiltered = seriesnew.filter((d) => {
          if (!d.date) return false;
          return (
            (!startDate || d.date >= startDate) &&
            (!endDate || d.date <= endDate)
          );
        });
      }
      if (!selectedEvent) {
        const yMax = d3.max(seriesFiltered, (d) => d.rate);
        const chart = resize((chartWidth) =>
          Plot.plot({
            width: chartWidth,
            height:
              margin.top +
              margin.bottom +
              baseMinRow +
              PX_PADDING +
              mobileExtra +
              9,
            y: { grid: true, label: '', domain: [0, yMax] },
            marks: [
              Plot.lineY(seriesFiltered, {
                x: 'date',
                y: 'rate',
                curve: 'step',
                tip: true,
                title: (d) =>
                  `Date: ${fmtYMD(d.date)}\n` +
                  `Value: ${d.rate != null ? d.rate.toFixed(2) : 'N/A'}`,
              }),
            ],
          }),
        );
        bodyEl.append(chart);
        return;
      }

      const s = selectedEvent.startDate || selectedEvent.date;
      const e =
        selectedEvent.endDate || selectedEvent.startDate || selectedEvent.date;
      const x0 = new Date(s.getTime() - 60 * DAY);
      const x1 = new Date(e.getTime() + 1 * DAY);
      const slice = seriesFiltered.filter((d) => d.date >= x0 && d.date <= x1);

      const series = slice.length ? slice : seriesFiltered;

      const yMin = d3.min(series, (d) => d.rate);
      const yMax = d3.max(series, (d) => d.rate);
      const marks = [
        Plot.lineY(series, {
          x: 'date',
          y: 'rate',
          curve: 'step',
          tip: true,
          title: (d) =>
            `Date: ${fmtYMD(d.date)}\n` +
            `Value: ${d.rate != null ? d.rate.toFixed(2) : 'N/A'}`,
        }),
        Plot.rectY([{ s, e }], {
          x1: (d) => d.s,
          x2: (d) => d.e,
          y1: yMin,
          y2: yMax,
          fill: '#ef4444',
          fillOpacity: 0.15,
          title: `${selectedEvent.dateLabel}\n${selectedEvent.who || ''}\n${selectedEvent.description || ''}`,
        }),
        Plot.ruleX([s], {
          stroke: '#ef4444',
          strokeOpacity: 0.9,
          strokeWidth: 2,
        }),
      ];
      if (+e !== +s)
        marks.push(
          Plot.ruleX([e], {
            stroke: '#ef4444',
            strokeOpacity: 0.9,
            strokeWidth: 2,
          }),
        );

      const chart = resize((chartWidth) =>
        Plot.plot({
          width: chartWidth,
          height:
            margin.top +
            margin.bottom +
            VISIBLE_ROWS * 90 +
            PX_PADDING +
            mobileExtra,
          y: { grid: true, label: '', domain: [0, yMax] },
          x: { domain: [x0, x1], nice: false },
          marks,
        }),
      );

      const meta = html`<div class="event-meta download-links"></div>`;
      bodyEl.append(chart, meta);

      const impactLabels = ['Low', 'Moderate', 'High', 'Severe'];
      const level = impactLabels[selectedEvent.impactQuartile ?? 0];
      const exportLink = html`<span>Download Chart: <a href="#">PNG</a></span>`;
      exportLink.querySelector('a').onclick = async (clickEvent) => {
        clickEvent.preventDefault();
        const svg = bodyEl.querySelector('svg');
        if (!svg) return;
        await exportChartPng({
          chart: svg,
          logoUrl,
          title: `${name} — ${selectedEvent.dateLabel}`,
          meta: [
            ['Country', name],
            ['Date', selectedEvent.dateLabel],
            ['Impact score', selectedEvent.title || '—'],
            ['Level', level],
          ],
          notes: selectedEvent.description || '',
          notesLabel: 'Context',
          filename: `cp-cenalert-event-${name.replace(/[^\w-]+/g, '-')}-${
            selectedEvent.startISO ?? fmtYMD(selectedEvent.date)
          }`,
        });
      };
      meta.append(exportLink);
    }

    renderRight(countryEvents.length ? countryEvents[0] : null);
    scroller.append(svg.node());

    function layoutNodes() {
      node.each(function (d) {
        const foEl = d3.select(this).select('foreignObject');
        const div = d3.select(this).select('.label-html').node();
        const labelH = Math.ceil(div?.scrollHeight || 0);
        const rowH = Math.max(baseMinRow, labelH + 50);
        d.__rowH = rowH;
        foEl.attr('height', rowH);
      });

      let yCursor = topOffset;
      node.each(function (d) {
        d.__y = yCursor;
        d3.select(this).attr('transform', `translate(${laneX}, ${d.__y})`);
        const halfGap = nodeGap / 2;
        const tileTop = -r - halfGap / 4;

        const tileHeight = 2 * r + halfGap / 2 - 4;

        d3.select(this)
          .select('.event-tile')
          .attr('x', -r - 30)
          .attr('y', tileTop + 2)
          .attr('width', labelDx + 1.5 * labelWidth)
          .attr('height', tileHeight);
        yCursor += d.__rowH + nodeGap;
      });

      const newBase = yCursor;
      height = newBase + r * 0.6;
      svg.attr('viewBox', `0 0 ${width} ${height}`);
      svg.style('height', 'auto');
      svg.style('max-height', 'none');
    }

    function doResizeLayout() {
      currentWidth = graphWrap.clientWidth || currentWidth;
      const sizes = computeSizes();
      r = sizes.r;
      baseMinRow = sizes.baseMinRow;
      nodeGap = sizes.nodeGap;
      topOffset = sizes.topOffset;

      width = graphWrap.clientWidth || width || 800;
      laneX = margin.left + r;

      svg.attr('viewBox', `0 0 ${width} ${height}`);

      labelDx = r + 12;
      label.attr('transform', `translate(${labelDx}, 0)`);
      fo.attr('x', labelDx);

      node.select('circle').attr('r', r);
      node
        .selectAll('text')
        .nodes()
        .forEach((t) => {
          d3.select(t).style('font-size', `${Math.round(r * 0.35)}px`);
        });

      fo.attr('y', r * 0.1);
      computeLabelWidth();

      requestAnimationFrame(() => {
        layoutNodes();
        if (labelWidth <= 220) {
          computeLabelWidth();
          requestAnimationFrame(layoutNodes);
        }
      });
    }

    computeLabelWidth();
    requestAnimationFrame(() => {
      layoutNodes();
      if (labelWidth <= 200) {
        computeLabelWidth();
        requestAnimationFrame(layoutNodes);
      }
    });

    const onResize = () => {
      doResizeLayout();
    };

    window.removeEventListener('resize', onResize);
    window.addEventListener('resize', onResize, { passive: true });
  }

  function openDetail(
    code,
    name,
    fullseries,
    timeseries,
    countryHasAnyEvents,
    selectedEventKey,
  ) {
    const fmtYMDdots = d3.utcFormat('%Y.%m.%d');
    const startDate = d3.min(timeseries, (d) => d.date);
    const endDate = d3.max(timeseries, (d) => d.date);

    const rawEvents = events.filter(
      (d) => String(d.country).toUpperCase() === code,
    );
    const impacts = rawEvents
      .map((d) => Number(d.impact))
      .filter((n) => Number.isFinite(n))
      .sort(d3.ascending);

    const q1 = d3.quantile(impacts, 0.25) ?? 0;
    const q2 = d3.quantile(impacts, 0.5) ?? 0;
    const q3 = d3.quantile(impacts, 0.75) ?? 0;

    const countryEvents = rawEvents
      .map((d) => {
        const nImpact = Number(d.impact);
        const start = d.startDate ? fmtYMDdots(new Date(d.startDate)) : null;
        const end = d.endDate ? fmtYMDdots(new Date(d.endDate)) : null;
        const dateLabel =
          start && end && start !== end
            ? `${start} - ${end}`
            : start || end || '—';

        // Determine Quartile (0=Low, 1=Moderate, 2=High, 3=Severe)
        let q = 0;
        if (nImpact >= q3) q = 3;
        else if (nImpact >= q2) q = 2;
        else if (nImpact >= q1) q = 1;
        else q = 0;

        return {
          date: new Date(d.peak || d.start),
          dateLabel,
          startISO: d.startDate || null,
          endISO: d.endDate || null,
          startDate: d.startDate ? new Date(d.startDate) : null,
          endDate: d.endDate ? new Date(d.endDate) : null,
          code: 'Impact',
          title: Number.isFinite(nImpact)
            ? formatImpact(nImpact).replace(/,/g, '.')
            : '—',
          who: d.reportedBy || '',
          impact: nImpact,
          impactQuartile: q,
          description: d.description || '',
          eventTitle: d.title,
          category: d.category,
          cause: d.cause,
          justification: d.justification,
          affectedServices: d.affectedServices,
          sources: d.sources,
          __matchKey: d.startDate
            ? String(d.startDate)
            : d.peak
              ? String(d.peak)
              : dateLabel,
        };
      })
      .filter((d) => {
        const s = d.startDate || d.date;
        const e = d.endDate || d.date;
        if (!s || !e) return false;
        return (!startDate || e >= startDate) && (!endDate || s <= endDate);
      })
      .sort((a, b) => b.date - a.date || a.title.localeCompare(b.title));

    deps.seriesForSelectedCountry = fullseries;
    renderDetail(code, name, countryEvents, countryHasAnyEvents);
    detailSection.hidden = false;

    if (selectedEventKey) {
      const nodes = Array.from(detailSection.querySelectorAll('.node'));
      let targetNode = null;

      for (const n of nodes) {
        const d = d3.select(n).datum();
        if (!d) continue;
        if (d.__matchKey && String(d.__matchKey) === String(selectedEventKey)) {
          targetNode = n;
          break;
        }
        if (d.dateLabel && d.dateLabel === selectedEventKey) {
          targetNode = n;
          break;
        }
      }

      if (targetNode) {
        const scroller = detailSection.querySelector('.events-scroller');
        if (scroller && typeof targetNode.scrollIntoView === 'function') {
          targetNode.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        targetNode.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      } else {
        const first = detailSection.querySelector('.node');
        if (first)
          first.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      }
    }
  }
  return openDetail;
}
