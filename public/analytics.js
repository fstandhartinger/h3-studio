const counter = document.querySelector('#visitor-count');

if (counter) {
  fetch('/api/analytics/visitors', { headers: { Accept: 'application/json' } })
    .then((response) => (response.ok ? response.json() : null))
    .then((data) => {
      if (Number.isSafeInteger(data?.visitors) && data.visitors >= 0) {
        counter.textContent = `${data.visitors.toLocaleString()} visitors`;
        counter.hidden = false;
      }
    })
    .catch(() => {});
}
