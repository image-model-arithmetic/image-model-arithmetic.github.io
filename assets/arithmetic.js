const $ = selector => document.querySelector(selector);
const buttons = selector => [...document.querySelectorAll(selector)];
for (const button of buttons('[data-image]')) {
  button.addEventListener('click', () => {
    const clean = button.dataset.image === 'clean';
    for (const item of buttons('[data-image]')) item.setAttribute('aria-pressed', String(item === button));
    $('#input-digit').src = `assets/source-${clean ? '1431' : '1615'}.png`;
    $('#input-digit').alt = clean ? 'Clean zero used as the next input' : 'Generated zero used as the next input';
    $('#output-digit').src = `assets/source-${clean ? '1654' : '1681'}.png`;
    $('#output-digit').alt = `Model B’s generated answer, read as ${clean ? 'eight' : 'eighteen'}`;
    $('#answer-status').classList.toggle('correct', clean);
    $('#answer-status strong').textContent = clean ? '8 · correct' : '18 · incorrect';
  });
}

const percentage = (n, d) => (100 * n / d).toFixed(1) + '%';
let data, current = 'mod23';
function selectCell(row, column) {
  const experiment = data.matrices[current];
  const item = experiment.cells[row][column];
  for (const button of buttons('#exchange-matrix button')) button.setAttribute('aria-pressed', String(+button.dataset.row === row && +button.dataset.column === column));
  $('#cell-title').textContent = `${experiment.models[row]} → ${experiment.models[column]}`;
  if (current === 'appearance') {
    $('#cell-result').textContent = `${item > 0 ? '+' : ''}${item} / 16`;
    $('#cell-description').textContent = `After converting the same drawings to black and white: ${data.matrices.binary.cells[row][column]} → ${data.matrices.binary.cells[row][column] + item} correct answers out of 16.`;
  } else {
    const [correct, total] = current === 'mod23' ? item : [item, 16];
    $('#cell-result').textContent = `${correct} / ${total}`;
    $('#cell-description').textContent = `${percentage(correct, total)}. ${experiment.cell_description}`;
  }
  $('#cell-kicker').textContent = row === column ? 'Same-model control' : 'Different training runs';
}
function renderMatrix(key) {
  current = key;
  const experiment = data.matrices[key];
  for (const button of buttons('[data-matrix]')) button.setAttribute('aria-pressed', String(button.dataset.matrix === key));
  $('#matrix-explanation').textContent = experiment.explanation;
  $('#matrix-total').textContent = experiment.total_description;
  const table = $('#exchange-matrix');
  table.classList.toggle('binary', key !== 'mod23');
  const header = document.createElement('tr');
  for (const name of ['Source ↓', ...experiment.models]) {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = name; header.append(th);
  }
  table.tHead.replaceChildren(header);
  const rows = experiment.models.map((name, row) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th'); th.scope = 'row'; th.textContent = name; tr.append(th);
    experiment.cells[row].forEach((item, column) => {
      const td = document.createElement('td'); const button = document.createElement('button');
      button.type = 'button'; button.dataset.row = row; button.dataset.column = column; button.dataset.self = row === column;
      let label;
      if (key === 'appearance') {
        label = `${item > 0 ? '+' : ''}${item}`;
        button.style.backgroundColor = item > 0 ? '#d4e7f8' : item < 0 ? '#f3d9d6' : '#f1f3f6';
      } else {
        const [n, d] = key === 'mod23' ? item : [item, 16]; label = `${n}/${d}`;
        button.style.backgroundColor = n / d === 1 ? '#d8e9f8' : n / d >= .95 ? '#e5eff7' : '#f3e8df';
      }
      button.textContent = label;
      button.setAttribute('aria-label', `${name} to ${experiment.models[column]}: ${label}${row === column ? ', same-model control' : ''}`);
      button.addEventListener('click', () => selectCell(row, column)); td.append(button); tr.append(td);
    });
    return tr;
  });
  table.tBodies[0].replaceChildren(...rows);
  selectCell(key === 'mod23' ? 0 : 1, key === 'mod23' ? 1 : 0);
}
try {
  const response = await fetch('data/results.json');
  if (!response.ok) throw new Error(`Results request returned ${response.status}`);
  data = await response.json();
  for (const button of buttons('[data-matrix]')) button.addEventListener('click', () => renderMatrix(button.dataset.matrix));
  renderMatrix('mod23');
} catch (error) {
  $('#matrix-explanation').textContent = 'The interactive results could not load. The complete results remain available in the paper and downloadable data.';
  for (const button of buttons('[data-matrix]')) button.disabled = true;
  console.error(error);
}
