async function request(path, params = {}) {
  const query = new URLSearchParams(params);
  const response = await fetch(`/api/${path}?${query}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`CenAlert API: HTTP ${response.status}`);
  return response.json();
}

export async function fetchCenalertCountries() {
  return (await request('countries')).map((d) => d.country);
}

async function records(path, { country, range } = {}) {
  const params = country ? { country } : {};
  if (range && typeof range === 'object') {
    if (range.startDate) params.start = range.startDate;
    if (range.endDate) params.end = range.endDate;
  }
  const result = [];
  for (let offset = 0; ; offset += 20000) {
    const page = await request(path, { ...params, offset, limit: 20000 });
    result.push(...page);
    if (page.length < 20000) return result;
  }
}

export const fetchCenalertEvents = (options) => records('events', options);
export const fetchCenalertTimeseries = (options) => records('series', options);
