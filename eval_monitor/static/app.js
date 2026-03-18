const state = {
  snapshot: null,
  runDetail: null,
  selectedSeriesId: null,
  selectedRunId: null,
  filters: {
    environment: "all",
    algorithm: "all",
    status: "all",
    search: "",
  },
  metric: "weighted_success_rate",
  stream: null,
  streamState: "connecting",
  loadPromise: null,
  reloadQueued: false,
  detailVersion: null,
}

const els = {
  evalRoot: document.querySelector("#eval-root"),
  snapshotVersion: document.querySelector("#snapshot-version"),
  lastRefresh: document.querySelector("#last-refresh"),
  streamDot: document.querySelector("#stream-dot"),
  streamStatus: document.querySelector("#stream-status"),
  refreshBtn: document.querySelector("#refresh-btn"),
  environmentFilter: document.querySelector("#environment-filter"),
  algorithmFilter: document.querySelector("#algorithm-filter"),
  statusFilter: document.querySelector("#status-filter"),
  metricFilter: document.querySelector("#metric-filter"),
  searchInput: document.querySelector("#search-input"),
  statsGrid: document.querySelector("#stats-grid"),
  matrix: document.querySelector("#matrix"),
  seriesList: document.querySelector("#series-list"),
  timelineTitle: document.querySelector("#timeline-title"),
  timeline: document.querySelector("#timeline"),
  selectLatestRunBtn: document.querySelector("#select-latest-run-btn"),
  taskCompare: document.querySelector("#task-compare"),
  runTable: document.querySelector("#run-table"),
  runDetail: document.querySelector("#run-detail"),
  deleteRunBtn: document.querySelector("#delete-run-btn"),
}


function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}


function percent(value) {
  if (typeof value !== "number") {
    return "—"
  }
  return `${(value * 100).toFixed(1)}%`
}


function decimal(value, digits = 1) {
  if (typeof value !== "number") {
    return "—"
  }
  return value.toFixed(digits)
}


function toneForRate(value) {
  if (typeof value !== "number") {
    return "warn"
  }
  if (value >= 0.7) {
    return "good"
  }
  if (value >= 0.35) {
    return "warn"
  }
  return "bad"
}


function prettyDate(value) {
  if (!value) {
    return "—"
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date)
}


function metricLabel(metricKey) {
  switch (metricKey) {
    case "latest_success_rate":
      return "最新成功率"
    case "best_success_rate":
      return "最佳成功率"
    case "average_success_rate":
      return "平均成功率"
    default:
      return "加权成功率"
  }
}


function failureSummary(run) {
  const parts = []
  for (const [key, value] of Object.entries(run.failure_counts || {})) {
    parts.push(`${key.replaceAll("_", " ")}=${value}`)
  }
  if (!parts.length) {
    for (const [key, value] of Object.entries(run.failure_reason_counts || {})) {
      parts.push(`${key}=${value}`)
    }
  }
  return parts.length ? parts.join(" · ") : "—"
}


function emptyState(message) {
  return `<div class="empty-state">${escapeHtml(message)}</div>`
}


async function requestJson(url, options = {}) {
  const response = await fetch(url, options)
  const payload = await response.json()
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload
}


function upsertOptions(select, options, currentValue, withAllLabel) {
  const html = []
  if (withAllLabel) {
    html.push(`<option value="all">${escapeHtml(withAllLabel)}</option>`)
  }
  for (const option of options) {
    html.push(`<option value="${escapeHtml(option.id)}">${escapeHtml(option.label)}</option>`)
  }
  select.innerHTML = html.join("")
  if ([...select.options].some((option) => option.value === currentValue)) {
    select.value = currentValue
  } else {
    select.value = "all"
  }
}


function allRuns() {
  if (!state.snapshot) {
    return []
  }
  return state.snapshot.series.flatMap((series) => series.runs)
}


function findSeries(seriesId) {
  return state.snapshot?.series.find((series) => series.id === seriesId) || null
}


function findRun(runId) {
  return allRuns().find((run) => run.id === runId) || null
}


function seriesMatchesSearch(series, search) {
  if (!search) {
    return true
  }
  const haystack = [
    series.id,
    series.environment,
    series.environment_family,
    series.algorithm,
    series.label,
    ...series.per_task.map((task) => task.task_code || task.goal_name || task.task_id || ""),
  ]
    .join(" ")
    .toLowerCase()
  return haystack.includes(search)
}


function runMatchesFilters(run) {
  const search = state.filters.search.trim().toLowerCase()
  if (state.filters.environment !== "all" && run.environment !== state.filters.environment) {
    return false
  }
  if (state.filters.algorithm !== "all" && run.algorithm !== state.filters.algorithm) {
    return false
  }
  if (state.filters.status !== "all" && run.status !== state.filters.status) {
    return false
  }
  if (!search) {
    return true
  }
  const haystack = [
    run.id,
    run.environment,
    run.algorithm,
    run.run_name,
    failureSummary(run),
  ]
    .join(" ")
    .toLowerCase()
  return haystack.includes(search)
}


function filteredSeries() {
  if (!state.snapshot) {
    return []
  }
  const search = state.filters.search.trim().toLowerCase()
  return state.snapshot.series.filter((series) => {
    if (state.filters.environment !== "all" && series.environment !== state.filters.environment) {
      return false
    }
    if (state.filters.algorithm !== "all" && series.algorithm !== state.filters.algorithm) {
      return false
    }
    if (state.filters.status !== "all") {
      const hasStatus = series.runs.some((run) => run.status === state.filters.status)
      if (!hasStatus) {
        return false
      }
    }
    return seriesMatchesSearch(series, search)
  })
}


function filteredRuns() {
  return allRuns()
    .filter(runMatchesFilters)
    .sort((left, right) => (right.last_modified_epoch || 0) - (left.last_modified_epoch || 0))
}


function syncSelections() {
  const visibleSeries = filteredSeries()
  const visibleRuns = filteredRuns()
  const visibleSeriesIds = new Set(visibleSeries.map((series) => series.id))
  const visibleRunIds = new Set(visibleRuns.map((run) => run.id))

  if (!state.selectedSeriesId || !visibleSeriesIds.has(state.selectedSeriesId)) {
    state.selectedSeriesId = visibleSeries[0]?.id || state.snapshot?.series[0]?.id || null
  }

  if (state.selectedRunId && !findRun(state.selectedRunId)) {
    state.selectedRunId = null
    state.runDetail = null
    state.detailVersion = null
  }

  if (!state.selectedRunId || !visibleRunIds.has(state.selectedRunId)) {
    const selectedSeries = findSeries(state.selectedSeriesId)
    state.selectedRunId =
      visibleRuns.find((run) => run.series_id === state.selectedSeriesId)?.id ||
      selectedSeries?.latest_complete_run_id ||
      selectedSeries?.latest_run_id ||
      visibleRuns[0]?.id ||
      null
  }

  if (state.selectedRunId) {
    const selectedRun = findRun(state.selectedRunId)
    if (selectedRun) {
      state.selectedSeriesId = selectedRun.series_id
    }
  }

  if (state.runDetail && state.runDetail.run.id !== state.selectedRunId) {
    state.runDetail = null
    state.detailVersion = null
  }
}


async function loadSnapshot({ force = false } = {}) {
  if (state.loadPromise) {
    state.reloadQueued = true
    return state.loadPromise
  }

  state.loadPromise = (async () => {
    const url = force ? `/api/data?ts=${Date.now()}` : "/api/data"
    const snapshot = await requestJson(url)
    state.snapshot = snapshot

    els.evalRoot.textContent = snapshot.eval_root
    els.snapshotVersion.textContent = `v${snapshot.version}`
    els.lastRefresh.textContent = prettyDate(snapshot.generated_at)

    upsertOptions(els.environmentFilter, snapshot.filters.environments, state.filters.environment, "全部环境")
    upsertOptions(els.algorithmFilter, snapshot.filters.algorithms, state.filters.algorithm, "全部算法")
    upsertOptions(els.statusFilter, snapshot.filters.statuses, state.filters.status, "全部状态")

    state.filters.environment = els.environmentFilter.value
    state.filters.algorithm = els.algorithmFilter.value
    state.filters.status = els.statusFilter.value
    syncSelections()
    state.runDetail = null
    state.detailVersion = null
    renderAll()
    await ensureRunDetail()
  })()

  try {
    await state.loadPromise
  } finally {
    state.loadPromise = null
    if (state.reloadQueued) {
      state.reloadQueued = false
      loadSnapshot({ force: true })
    }
  }
}


async function ensureRunDetail() {
  if (!state.selectedRunId) {
    state.runDetail = null
    state.detailVersion = null
    renderRunDetail()
    return
  }
  if (state.runDetail?.run?.id === state.selectedRunId && state.detailVersion === state.snapshot?.version) {
    renderRunDetail()
    return
  }
  const detail = await requestJson(`/api/run?id=${encodeURIComponent(state.selectedRunId)}`)
  if (state.selectedRunId === detail.run.id) {
    state.runDetail = detail
    state.detailVersion = state.snapshot?.version || null
    renderRunDetail()
  }
}


function renderAll() {
  renderStats()
  renderMatrix()
  renderSeriesList()
  renderTimeline()
  renderTaskCompare()
  renderRunTable()
  renderRunDetail()
}


function renderStats() {
  if (!state.snapshot) {
    els.statsGrid.innerHTML = ""
    return
  }
  const { stats } = state.snapshot
  const cards = [
    ["Series", stats.series_count, `${stats.environment_count} env · ${stats.algorithm_count} alg`],
    ["Runs", stats.run_count, `${stats.video_count} videos indexed`],
    ["Complete", stats.complete_runs, "有 summary 且可比较"],
    ["Incomplete", stats.incomplete_runs, "进行中或中断的目录"],
    ["Rollouts", stats.total_rollouts, "完整 run 的总 rollout 数"],
  ]
  els.statsGrid.innerHTML = cards
    .map(
      ([label, value, meta]) => `
        <article class="stat-card">
          <p class="stat-card__label">${escapeHtml(label)}</p>
          <p class="stat-card__value">${escapeHtml(value)}</p>
          <p class="stat-card__meta">${escapeHtml(meta)}</p>
        </article>
      `
    )
    .join("")
}


function renderMatrix() {
  if (!state.snapshot) {
    els.matrix.innerHTML = ""
    return
  }
  const visibleEnvironment = state.filters.environment
  const visibleAlgorithm = state.filters.algorithm
  const algorithms = state.snapshot.matrix.algorithms.filter(
    (algorithm) => visibleAlgorithm === "all" || algorithm === visibleAlgorithm
  )
  const rows = state.snapshot.matrix.rows.filter(
    (row) => visibleEnvironment === "all" || row.environment === visibleEnvironment
  )

  if (!algorithms.length || !rows.length) {
    els.matrix.innerHTML = emptyState("当前筛选下没有可比较的环境/算法组合。")
    return
  }

  const metricKey = state.metric
  const table = `
    <table class="matrix-table">
      <thead>
        <tr>
          <th>Environment</th>
          ${algorithms.map((algorithm) => `<th>${escapeHtml(algorithm)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const cells = algorithms
              .map((algorithm) => {
                const cell = row.cells[algorithm]
                if (!cell) {
                  return `
                    <td>
                      <div class="matrix-cell is-empty">
                        <div class="subtle">No series</div>
                      </div>
                    </td>
                  `
                }
                const metric = cell[metricKey]
                const selected = cell.series_id === state.selectedSeriesId ? "is-selected" : ""
                return `
                  <td>
                    <div class="matrix-cell ${selected}" data-series-id="${escapeHtml(cell.series_id)}">
                      <div class="metric-value" data-tone="${toneForRate(metric)}">${percent(metric)}</div>
                      <div class="subtle">${metricLabel(metricKey)}</div>
                      <div class="chip-row" style="margin-top:10px;">
                        <span class="chip chip--success">complete ${cell.complete_runs}</span>
                        <span class="chip chip--warning">pending ${cell.incomplete_runs}</span>
                      </div>
                      <div class="subtle" style="margin-top:10px;">rollouts ${cell.total_rollouts || 0}</div>
                    </div>
                  </td>
                `
              })
              .join("")
            return `
              <tr>
                <td>
                  <div class="series-card__title">${escapeHtml(row.label)}</div>
                  <div class="series-card__subtitle">${escapeHtml(row.family)}</div>
                </td>
                ${cells}
              </tr>
            `
          })
          .join("")}
      </tbody>
    </table>
  `
  els.matrix.innerHTML = table
  els.matrix.querySelectorAll("[data-series-id]").forEach((node) => {
    node.addEventListener("click", () => {
      selectSeries(node.dataset.seriesId, { selectLatestRun: true })
    })
  })
}


function renderSeriesList() {
  const seriesList = filteredSeries()
  if (!seriesList.length) {
    els.seriesList.innerHTML = emptyState("当前筛选下没有实验序列。")
    return
  }

  els.seriesList.innerHTML = seriesList
    .map((series) => {
      const selected = series.id === state.selectedSeriesId ? "is-selected" : ""
      const metric = series[state.metric]
      return `
        <article class="series-card ${selected}" data-series-card="${escapeHtml(series.id)}">
          <div class="series-card__top">
            <div>
              <div class="series-card__title">${escapeHtml(series.environment)}</div>
              <div class="series-card__subtitle">${escapeHtml(series.algorithm)}</div>
            </div>
            <button class="button" data-delete-series="${escapeHtml(series.id)}">删除序列</button>
          </div>
          <div class="metric-row">
            <div>
              <div class="metric-value" data-tone="${toneForRate(metric)}">${percent(metric)}</div>
              <div class="subtle">${metricLabel(state.metric)}</div>
            </div>
            <div class="subtle">
              <div>latest ${percent(series.latest_success_rate)}</div>
              <div>best ${percent(series.best_success_rate)}</div>
            </div>
          </div>
          <div class="chip-row" style="margin-top:12px;">
            <span class="chip chip--success">complete ${series.complete_runs}</span>
            <span class="chip chip--warning">incomplete ${series.incomplete_runs}</span>
            ${series.invalid_runs ? `<span class="chip chip--danger">invalid ${series.invalid_runs}</span>` : ""}
          </div>
          <div class="series-card__bottom" style="margin-top:12px;">
            <div class="subtle">rollouts ${series.total_rollouts || 0}</div>
            <div class="subtle">${escapeHtml(prettyDate(series.last_updated))}</div>
          </div>
        </article>
      `
    })
    .join("")

  els.seriesList.querySelectorAll("[data-series-card]").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest("[data-delete-series]")) {
        return
      }
      selectSeries(card.dataset.seriesCard, { selectLatestRun: false })
    })
  })

  els.seriesList.querySelectorAll("[data-delete-series]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation()
      const seriesId = button.dataset.deleteSeries
      const series = findSeries(seriesId)
      if (!series) {
        return
      }
      const confirmed = window.confirm(
        `删除整个实验序列？\n${series.id}\n这会移除该目录下的所有 run。`
      )
      if (!confirmed) {
        return
      }
      await requestJson("/api/delete-series", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: seriesId }),
      })
      if (state.selectedSeriesId === seriesId) {
        state.selectedSeriesId = null
        state.selectedRunId = null
        state.runDetail = null
        state.detailVersion = null
      }
      await loadSnapshot({ force: true })
    })
  })
}


function renderTimeline() {
  const series = findSeries(state.selectedSeriesId)
  if (!series) {
    els.timelineTitle.textContent = "Run 时间线"
    els.timeline.innerHTML = emptyState("选择一个实验序列后，这里会显示每轮 run 的走势。")
    return
  }

  els.timelineTitle.textContent = `${series.environment} / ${series.algorithm} 时间线`
  const completeRuns = [...series.runs]
    .filter((run) => run.status === "complete" && typeof run.success_rate === "number")
    .sort((left, right) => (left.timestamp_epoch || 0) - (right.timestamp_epoch || 0))
  const incompleteRuns = series.runs.filter((run) => run.status !== "complete")

  if (!completeRuns.length && !incompleteRuns.length) {
    els.timeline.innerHTML = emptyState("当前序列没有 run 数据。")
    return
  }

  if (!completeRuns.length) {
    els.timeline.innerHTML = `
      ${emptyState("当前只有 incomplete / invalid run，暂时无法绘制成功率曲线。")}
      <div class="chip-row" style="margin-top:12px;">
        ${incompleteRuns
          .map((run) => `<span class="chip chip--warning">${escapeHtml(run.run_name)} · ${escapeHtml(run.status)}</span>`)
          .join("")}
      </div>
    `
    return
  }

  const width = 920
  const height = 320
  const padding = { top: 22, right: 24, bottom: 56, left: 54 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom
  const xStep = completeRuns.length > 1 ? chartWidth / (completeRuns.length - 1) : chartWidth / 2

  const points = completeRuns.map((run, index) => {
    const x = padding.left + (completeRuns.length > 1 ? xStep * index : chartWidth / 2)
    const y = padding.top + chartHeight * (1 - run.success_rate)
    return { x, y, run }
  })
  const polyline = points.map((point) => `${point.x},${point.y}`).join(" ")
  const yTicks = [0, 0.25, 0.5, 0.75, 1]

  const svg = `
    <svg class="timeline-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="run timeline">
      ${yTicks
        .map((tick) => {
          const y = padding.top + chartHeight * (1 - tick)
          return `
            <line class="timeline-grid" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" />
            <text class="timeline-axis" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}%</text>
          `
        })
        .join("")}
      <polyline class="timeline-line" points="${polyline}" />
      ${points
        .map((point) => {
          const selected = point.run.id === state.selectedRunId ? "is-selected" : ""
          return `
            <circle
              class="timeline-point ${selected}"
              data-run-id="${escapeHtml(point.run.id)}"
              cx="${point.x}"
              cy="${point.y}"
              r="7"
            />
            <text class="timeline-axis" x="${point.x}" y="${height - 22}" text-anchor="middle">${escapeHtml(point.run.run_name.slice(5))}</text>
          `
        })
        .join("")}
    </svg>
  `

  const partial = incompleteRuns.length
    ? `
      <div class="chip-row" style="margin-top:12px;">
        ${incompleteRuns
          .map((run) => `<span class="chip chip--warning">${escapeHtml(run.run_name)} · ${escapeHtml(run.status)} · videos ${run.video_count}</span>`)
          .join("")}
      </div>
    `
    : ""

  els.timeline.innerHTML = svg + partial
  els.timeline.querySelectorAll("[data-run-id]").forEach((node) => {
    node.addEventListener("click", () => {
      selectRun(node.dataset.runId)
    })
  })
}


function renderTaskCompare() {
  if (!state.snapshot) {
    els.taskCompare.innerHTML = ""
    return
  }

  const series = findSeries(state.selectedSeriesId)
  const environment = series?.environment || (state.filters.environment !== "all" ? state.filters.environment : null)
  if (!environment) {
    els.taskCompare.innerHTML = emptyState("先选择一个环境或实验序列。")
    return
  }

  const compareSeries = state.snapshot.series
    .filter((item) => item.environment === environment)
    .filter((item) => state.filters.algorithm === "all" || item.algorithm === state.filters.algorithm)

  if (!compareSeries.length) {
    els.taskCompare.innerHTML = emptyState("当前环境没有可比较的算法结果。")
    return
  }

  const algorithms = compareSeries.map((item) => item.algorithm)
  const taskMap = new Map()
  for (const item of compareSeries) {
    for (const task of item.per_task) {
      const key = task.task_code || task.goal_name || task.task_id || task.task_key
      if (!taskMap.has(key)) {
        taskMap.set(key, {
          label: task.task_code || task.goal_name || task.task_id || task.task_key,
          goal_name: task.goal_name || "",
          values: {},
        })
      }
      taskMap.get(key).values[item.algorithm] = task
    }
  }

  const rows = [...taskMap.values()]
  rows.sort((left, right) => left.label.localeCompare(right.label, "zh-CN"))

  els.taskCompare.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Task</th>
          ${algorithms.map((algorithm) => `<th>${escapeHtml(algorithm)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const cells = algorithms
              .map((algorithm) => {
                const task = row.values[algorithm]
                if (!task) {
                  return "<td class='subtle'>—</td>"
                }
                return `
                  <td>
                    <div class="metric-value" data-tone="${toneForRate(task.success_rate)}" style="font-size:1.1rem;">${percent(task.success_rate)}</div>
                    <div class="subtle">${task.success_count}/${task.rollouts}</div>
                  </td>
                `
              })
              .join("")
            return `
              <tr>
                <td>
                  <div class="series-card__title">${escapeHtml(row.label)}</div>
                  <div class="series-card__subtitle">${escapeHtml(row.goal_name)}</div>
                </td>
                ${cells}
              </tr>
            `
          })
          .join("")}
      </tbody>
    </table>
  `
}


function renderRunTable() {
  const runs = filteredRuns()
  if (!runs.length) {
    els.runTable.innerHTML = emptyState("当前筛选下没有 run。")
    return
  }

  els.runTable.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Environment</th>
          <th>Algorithm</th>
          <th>Status</th>
          <th>Rollouts</th>
          <th>Success</th>
          <th>Failures</th>
        </tr>
      </thead>
      <tbody>
        ${runs
          .map((run) => {
            const selected = run.id === state.selectedRunId ? "is-selected" : ""
            return `
              <tr class="run-row ${selected}" data-run-row="${escapeHtml(run.id)}">
                <td>
                  <div class="series-card__title">${escapeHtml(run.run_name)}</div>
                  <div class="series-card__subtitle">${escapeHtml(prettyDate(run.timestamp))}</div>
                </td>
                <td>${escapeHtml(run.environment)}</td>
                <td>${escapeHtml(run.algorithm)}</td>
                <td><span class="chip ${run.status === "complete" ? "chip--success" : run.status === "invalid" ? "chip--danger" : "chip--warning"}">${escapeHtml(run.status)}</span></td>
                <td>${escapeHtml(run.num_rollouts ?? run.video_count)}</td>
                <td>
                  <div class="metric-value" data-tone="${toneForRate(run.success_rate)}" style="font-size:1.05rem;">${percent(run.success_rate)}</div>
                </td>
                <td class="subtle">${escapeHtml(failureSummary(run))}</td>
              </tr>
            `
          })
          .join("")}
      </tbody>
    </table>
  `

  els.runTable.querySelectorAll("[data-run-row]").forEach((row) => {
    row.addEventListener("click", () => {
      selectRun(row.dataset.runRow)
    })
  })
}


function renderRunDetail() {
  const detail = state.runDetail
  els.deleteRunBtn.disabled = !detail

  if (!detail) {
    els.runDetail.innerHTML = emptyState("选择一个 run 后，这里会显示任务统计和 rollout 视频。")
    return
  }

  const run = detail.run
  const failureChips = Object.entries(run.failure_counts || {})
    .map(([key, value]) => `<span class="chip chip--warning">${escapeHtml(key)} ${value}</span>`)
    .join("")
  const reasonChips = Object.entries(run.failure_reason_counts || {})
    .map(([key, value]) => `<span class="chip chip--danger">${escapeHtml(key)} ${value}</span>`)
    .join("")

  const taskTable = detail.per_task.length
    ? `
      <table class="data-table">
        <thead>
          <tr>
            <th>Task</th>
            <th>Rollouts</th>
            <th>Success</th>
            <th>Rate</th>
            <th>Failures</th>
          </tr>
        </thead>
        <tbody>
          ${detail.per_task
            .map((task) => {
              const failures = Object.entries(task.failure_counts || {})
                .map(([key, value]) => `${key}=${value}`)
                .join(" · ")
              return `
                <tr>
                  <td>
                    <div class="series-card__title">${escapeHtml(task.task_code || task.task_id || task.task_key)}</div>
                    <div class="series-card__subtitle">${escapeHtml(task.goal_name || "")}</div>
                  </td>
                  <td>${escapeHtml(task.rollouts ?? "—")}</td>
                  <td>${escapeHtml(task.success_count ?? "—")}</td>
                  <td><span class="metric-value" data-tone="${toneForRate(task.success_rate)}" style="font-size:1.05rem;">${percent(task.success_rate)}</span></td>
                  <td class="subtle">${escapeHtml(failures || "—")}</td>
                </tr>
              `
            })
            .join("")}
        </tbody>
      </table>
    `
    : emptyState("没有 task 级统计。")

  const resultsHtml = detail.results.length
    ? `
      <div class="video-grid">
        ${detail.results
          .map((result) => {
            const label = result.task_code || result.target_goal_name || result.task_id || "unknown"
            const statusChip =
              result.success === true
                ? '<span class="chip chip--success">success</span>'
                : result.success === false
                  ? '<span class="chip chip--danger">failed</span>'
                  : '<span class="chip chip--warning">pending</span>'
            return `
              <article class="video-card">
                <div class="series-card__top">
                  <div>
                    <div class="series-card__title">episode ${escapeHtml(result.episode_index ?? "—")} · ${escapeHtml(label)}</div>
                    <div class="series-card__subtitle">${escapeHtml(result.target_goal_name || result.final_phase_name || "")}</div>
                  </div>
                  ${statusChip}
                </div>
                <div class="video-card__meta">
                  <div>steps: ${escapeHtml(result.steps ?? "—")} · reward: ${escapeHtml(decimal(result.sum_reward, 2))}</div>
                  <div>failure: ${escapeHtml(result.failure_reason || "—")} · collisions: ${escapeHtml(result.collision_rejections ?? "—")}</div>
                  <div class="mono">${escapeHtml(result.video_name)}</div>
                </div>
                <video controls preload="metadata" src="${escapeHtml(result.media_url)}"></video>
              </article>
            `
          })
          .join("")}
      </div>
    `
    : emptyState("没有 rollout 结果。")

  els.runDetail.innerHTML = `
    <section class="detail-section">
      <div class="series-card__top">
        <div>
          <div class="series-card__title">${escapeHtml(run.environment)} / ${escapeHtml(run.algorithm)}</div>
          <div class="series-card__subtitle">${escapeHtml(run.run_name)} · ${escapeHtml(run.status)}</div>
        </div>
        <div class="metric-value" data-tone="${toneForRate(run.success_rate)}">${percent(run.success_rate)}</div>
      </div>
      <div class="detail-grid" style="margin-top:14px;">
        <div class="detail-kv">
          <span class="detail-kv__label">Success</span>
          <span class="detail-kv__value">${escapeHtml(run.success_count ?? "—")} / ${escapeHtml(run.num_rollouts ?? run.video_count ?? "—")}</span>
        </div>
        <div class="detail-kv">
          <span class="detail-kv__label">Last Modified</span>
          <span class="detail-kv__value">${escapeHtml(prettyDate(run.last_modified))}</span>
        </div>
        <div class="detail-kv">
          <span class="detail-kv__label">Avg Steps</span>
          <span class="detail-kv__value">${escapeHtml(decimal(run.avg_steps, 1))}</span>
        </div>
        <div class="detail-kv">
          <span class="detail-kv__label">Avg Collision Rejections</span>
          <span class="detail-kv__value">${escapeHtml(decimal(run.avg_collision_rejections, 1))}</span>
        </div>
        <div class="detail-kv">
          <span class="detail-kv__label">Path</span>
          <span class="detail-kv__value mono">${escapeHtml(run.path)}</span>
        </div>
        <div class="detail-kv">
          <span class="detail-kv__label">Policy Dir</span>
          <span class="detail-kv__value mono">${escapeHtml(run.policy_dir || "—")}</span>
        </div>
      </div>
      ${(failureChips || reasonChips) ? `<div class="chip-row" style="margin-top:12px;">${failureChips}${reasonChips}</div>` : ""}
      ${run.summary_error ? `<p class="danger-link" style="margin:12px 0 0;">${escapeHtml(run.summary_error)}</p>` : ""}
    </section>
    <section class="detail-section">
      <div class="series-card__title">Per-task Breakdown</div>
      <div style="margin-top:10px;">${taskTable}</div>
    </section>
    <section class="detail-section">
      <div class="series-card__title">Rollout Videos</div>
      <div style="margin-top:10px;">${resultsHtml}</div>
    </section>
  `
}


function setStreamState(nextState, label) {
  state.streamState = nextState
  els.streamDot.dataset.state = nextState
  els.streamStatus.textContent = label
}


function connectStream() {
  if (state.stream) {
    state.stream.close()
  }

  setStreamState("connecting", "connecting")
  const stream = new EventSource("/api/stream")
  state.stream = stream

  stream.addEventListener("snapshot", () => {
    setStreamState("connected", "live")
  })

  stream.addEventListener("update", async () => {
    setStreamState("connected", "live")
    await loadSnapshot({ force: true })
  })

  stream.onerror = () => {
    setStreamState("disconnected", "reconnecting")
    stream.close()
    if (state.stream === stream) {
      state.stream = null
    }
    window.setTimeout(connectStream, 2500)
  }
}


async function selectSeries(seriesId, { selectLatestRun }) {
  state.selectedSeriesId = seriesId
  const series = findSeries(seriesId)
  const currentRun = findRun(state.selectedRunId)
  if (series && (selectLatestRun || currentRun?.series_id !== seriesId)) {
    state.selectedRunId = series.latest_complete_run_id || series.latest_run_id || null
    state.runDetail = null
    state.detailVersion = null
  }
  renderAll()
  await ensureRunDetail()
}


async function selectRun(runId) {
  const run = findRun(runId)
  state.selectedRunId = runId
  if (run) {
    state.selectedSeriesId = run.series_id
  }
  state.runDetail = null
  state.detailVersion = null
  renderAll()
  await ensureRunDetail()
}


function bindGlobalEvents() {
  els.refreshBtn.addEventListener("click", async () => {
    await requestJson("/api/refresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    await loadSnapshot({ force: true })
  })

  els.environmentFilter.addEventListener("change", async () => {
    state.filters.environment = els.environmentFilter.value
    syncSelections()
    renderAll()
    await ensureRunDetail()
  })

  els.algorithmFilter.addEventListener("change", async () => {
    state.filters.algorithm = els.algorithmFilter.value
    syncSelections()
    renderAll()
    await ensureRunDetail()
  })

  els.statusFilter.addEventListener("change", async () => {
    state.filters.status = els.statusFilter.value
    syncSelections()
    renderAll()
    await ensureRunDetail()
  })

  els.metricFilter.addEventListener("change", () => {
    state.metric = els.metricFilter.value
    renderAll()
  })

  els.searchInput.addEventListener("input", async () => {
    state.filters.search = els.searchInput.value
    syncSelections()
    renderAll()
    await ensureRunDetail()
  })

  els.selectLatestRunBtn.addEventListener("click", async () => {
    const series = findSeries(state.selectedSeriesId)
    if (!series) {
      return
    }
    const runId = series.latest_complete_run_id || series.latest_run_id
    if (runId) {
      await selectRun(runId)
    }
  })

  els.deleteRunBtn.addEventListener("click", async () => {
    if (!state.runDetail) {
      return
    }
    const runId = state.runDetail.run.id
    const confirmed = window.confirm(`删除当前 run？\n${runId}`)
    if (!confirmed) {
      return
    }
    await requestJson("/api/delete-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: runId }),
    })
    state.selectedRunId = null
    state.runDetail = null
    state.detailVersion = null
    await loadSnapshot({ force: true })
  })
}


async function bootstrap() {
  bindGlobalEvents()
  await loadSnapshot({ force: true })
  connectStream()
}


bootstrap().catch((error) => {
  console.error(error)
  els.runDetail.innerHTML = emptyState(`初始化失败: ${error.message}`)
})
