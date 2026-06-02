import { EXACT_POINT_CLOUD_DISCLAIMER, loadDemoBundle } from './data-loader'
import { DemoScene } from './demo-scene'
import type {
  DemoConfig,
  DemoRoute,
  MetricValue,
  NavigationGraphData,
  SceneStatus,
  SimulationScenario,
  StageDefinition,
} from './types'

const STAGES: StageDefinition[] = [
  {
    id: 1,
    label: 'Stage 1',
    title: 'BIM Input Model',
    description: 'Show the aligned station BIM as the starting context before any abstraction into navigation-ready data structures.',
    footnote: 'This is the raw geometric reference used to explain what later pipeline stages extract, sample, and connect.',
  },
  {
    id: 2,
    label: 'Stage 2',
    title: 'Geometry Extraction',
    description: 'Show the walkable geometry abstraction extracted from BIM: floor-supporting regions, vertical connectors, and navigation-relevant context.',
    footnote: 'This corresponds to pipeline STEP 1. The full BIM is replaced by a lightweight shell so the extracted method output reads clearly.',
  },
  {
    id: 3,
    label: 'Stage 3',
    title: 'Node Sampling',
    description: 'Show the valid navigation samples generated from the walkable geometry after clearance filtering, forbidden-zone removal, and connector snapping.',
    footnote: `${EXACT_POINT_CLOUD_DISCLAIMER} In the fallback path, dense samples are aggregated into readable tiles while preserving level density and method intent.`,
  },
  {
    id: 4,
    label: 'Stage 4',
    title: 'Navigation Graph Construction',
    description: 'Expose the 2.5D multilevel navigation graph built from sampled nodes and highlighted with fare gates, entrances, and vertical connectors.',
    footnote: 'This corresponds to pipeline STEP 3. Special nodes are highlighted directly in the scene so the graph-building logic is legible at a glance.',
  },
  {
    id: 5,
    label: 'Stage 5',
    title: 'Routing and OD Paths',
    description: 'Highlight representative paths computed on the graph and explain how OD pairs are snapped and solved with Dijkstra.',
    footnote: 'This corresponds to pipeline STEP 4. Route examples come from public/data/routes.json or an in-browser Dijkstra fallback.',
  },
  {
    id: 6,
    label: 'Stage 6',
    title: 'ABM Simulation',
    description: 'Play the agent-based simulation on top of the routed graph and surface congestion-driven behaviour over time.',
    footnote: 'This corresponds to pipeline STEP 5. Missing assets only trigger warnings and do not block the demonstrator.',
  },
  {
    id: 7,
    label: 'Stage 7',
    title: 'Integrated Evaluation View',
    description: 'Bring the BIM shell, graph, route, and simulation together as a high-level evaluation and narration view.',
    footnote: 'Use this stage as the pipeline summary after geometry extraction, sampling, graph construction, routing, and simulation have been explained.',
  },
]

interface StageMethodSummary {
  step: string
  input: string
  operation: string
  output: string
}

const STAGE_METHODS: Record<StageDefinition['id'], StageMethodSummary> = {
  1: {
    step: 'Context BIM',
    input: 'Aligned IFC-derived station model',
    operation: 'Expose the raw geometric reference before simplification',
    output: 'Whole-station BIM context',
  },
  2: {
    step: 'Pipeline STEP 1',
    input: 'IFC slabs, obstacles, doors, and connector tables',
    operation: 'Extract walkable floor support, connector endpoints, and navigation context',
    output: 'Lightweight geometry shell',
  },
  3: {
    step: 'Pipeline STEP 2',
    input: 'Walkable geometry and connector anchors',
    operation: 'Sample valid navigation nodes with clearance and forbidden-zone filtering',
    output: 'Readable node field / density tiles',
  },
  4: {
    step: 'Pipeline STEP 3',
    input: 'Sampled nodes plus cross-level connectors',
    operation: 'Connect neighbours, validate line of sight, and weight vertical traversal',
    output: '2.5D navigation graph',
  },
  5: {
    step: 'Pipeline STEP 4',
    input: 'Graph, semantic regions, and OD demand',
    operation: 'Compute Dijkstra paths and snap routes to the graph',
    output: 'Representative OD paths',
  },
  6: {
    step: 'Pipeline STEP 5',
    input: 'Graph, routes, and scenario demand',
    operation: 'Advance agents, record congestion, and replay scenario dynamics',
    output: 'Simulation trajectories and queues',
  },
  7: {
    step: 'Pipeline STEP 6',
    input: 'Routes, simulations, and graph metrics',
    operation: 'Summarize the pipeline outputs into an evaluation-oriented narrative',
    output: 'Integrated method overview',
  },
}

interface References {
  stageButtons: HTMLButtonElement[]
  stageLabel: HTMLElement
  stageTitle: HTMLElement
  stageDescription: HTMLElement
  stageFootnote: HTMLElement
  stageProgress: HTMLElement
  methodStep: HTMLElement
  methodGrid: HTMLElement
  stageOverlayLabel: HTMLElement
  stageOverlayTitle: HTMLElement
  stageOverlayDescription: HTMLElement
  stageOverlayProgress: HTMLElement
  metrics: HTMLElement
  routeSelect: HTMLSelectElement
  scenarioSelect: HTMLSelectElement
  sceneHost: HTMLElement
  loading: HTMLElement
  error: HTMLElement
  warningPanel: HTMLElement
  warnings: HTMLElement
  helpOverlay: HTMLElement
  legendPanel: HTMLElement
  pointCloudNote: HTMLElement
  prevButton: HTMLButtonElement
  nextButton: HTMLButtonElement
  resetButton: HTMLButtonElement
  autoRotateButton: HTMLButtonElement
  playButton: HTMLButtonElement
  helpButton: HTMLButtonElement
  fullscreenButton: HTMLButtonElement
  routeWrap: HTMLElement
  scenarioWrap: HTMLElement
  footerNote: HTMLElement
}

function buildShell(): string {
  return `
    <div class="demo-shell">
      <aside class="sidebar">
        <div class="brand-card">
          <div class="brand-meta">
            <span class="meta-pill">Offline-ready</span>
            <span class="meta-pill">7-stage narrative</span>
          </div>
          <p class="eyebrow">Interactive Demonstrator</p>
          <h1 class="brand-title">BIM to Navigation Digital Twin</h1>
          <p class="brand-copy" data-stage-description-static>
            Local thesis-defense demonstrator with deterministic stages, offline data loading, and keyboard or mouse review controls.
          </p>
        </div>

        <section class="panel stage-panel">
          <div class="panel-header">
            <div class="panel-header-row">
              <p class="eyebrow" data-stage-label></p>
              <span class="stage-progress-chip" data-stage-progress></span>
            </div>
            <h2 data-stage-title></h2>
          </div>
          <p class="panel-copy" data-stage-description></p>
          <p class="panel-footnote" data-stage-footnote></p>
          <div class="stage-button-grid" data-stage-buttons></div>
        </section>

        <section class="panel method-panel">
          <div class="panel-header compact">
            <p class="eyebrow" data-method-step></p>
            <h2>Method Breakdown</h2>
          </div>
          <div class="method-grid" data-method-grid></div>
        </section>

        <section class="panel control-panel">
          <div class="panel-header compact">
            <p class="eyebrow">Controls</p>
            <h2>Navigation and review</h2>
          </div>
          <div class="control-row">
            <button type="button" data-prev>Previous</button>
            <button type="button" data-next>Next</button>
          </div>
          <div class="control-row">
            <button type="button" data-play>Play simulation</button>
            <button type="button" data-reset>Reset view</button>
          </div>
          <div class="control-row">
            <button type="button" data-autorotate>Auto-rotate</button>
            <button type="button" data-fullscreen>Fullscreen</button>
          </div>
          <div class="control-row control-row-single">
            <button type="button" class="wide-button" data-help>Help</button>
          </div>
          <p class="control-note">Mouse and keyboard controls are active.</p>
        </section>

        <section class="panel selector-panel">
          <div class="panel-header compact">
            <p class="eyebrow">Inputs</p>
            <h2>Route and scenario</h2>
          </div>
          <div class="selector-wrap" data-route-wrap>
            <label for="route-select">Route example</label>
            <select id="route-select" data-route-select></select>
          </div>
          <div class="selector-wrap" data-scenario-wrap>
            <label for="scenario-select">Simulation scenario</label>
            <select id="scenario-select" data-scenario-select></select>
          </div>
          <p class="footer-note" data-footer-note></p>
        </section>

        <section class="panel metrics-panel">
          <div class="panel-header compact">
            <p class="eyebrow">Current Stage</p>
            <h2>Metrics and Labels</h2>
          </div>
          <div class="metric-grid" data-metrics></div>
        </section>

        <section class="panel warning-panel hidden" data-warning-panel>
          <div class="panel-header compact">
            <p class="eyebrow">Warnings</p>
            <h2>Non-blocking fallback notices</h2>
          </div>
          <div class="warning-list" data-warnings></div>
        </section>
      </aside>

      <main class="viewport-shell">
        <div class="viewport-frame">
          <div class="scene-host" data-scene-host></div>

          <div class="viewport-overlay stage-spotlight">
            <div class="stage-spotlight-meta">
              <p class="eyebrow" data-stage-overlay-label></p>
              <span class="stage-progress-chip" data-stage-overlay-progress></span>
            </div>
            <h2 data-stage-overlay-title></h2>
            <p data-stage-overlay-description></p>
          </div>

          <div class="viewport-overlay legend-panel hidden" data-legend>
            <p class="eyebrow">Graph cues</p>
            <div class="legend-grid">
              <div><span class="swatch swatch-graph"></span>Level graph backbone</div>
              <div><span class="swatch swatch-route"></span>Snapped route</div>
              <div><span class="swatch swatch-fare"></span>Fare gates</div>
              <div><span class="swatch swatch-entrance"></span>Entrances and exits</div>
              <div><span class="swatch swatch-vertical"></span>Stairs, escalators, elevators</div>
              <div><span class="swatch swatch-model"></span>Subdued BIM shell</div>
              <div><span class="swatch swatch-sim"></span>Simulation agents</div>
            </div>
          </div>

          <div class="viewport-overlay note-panel hidden" data-point-cloud-note>
            <p class="eyebrow">Point-cloud disclaimer</p>
            <p>${EXACT_POINT_CLOUD_DISCLAIMER}</p>
          </div>

          <div class="viewport-overlay loading-panel" data-loading>
            <div class="loading-card">
              <p class="eyebrow">Loading</p>
              <h2>Preparing thesis outputs</h2>
              <p>Reading configuration, converted graph data, route samples, and simulation frames.</p>
            </div>
          </div>

          <div class="viewport-overlay error-panel hidden" data-error></div>
        </div>
      </main>

      <div class="help-overlay hidden" data-help-overlay>
        <div class="help-card">
          <div class="help-header">
            <div>
              <p class="eyebrow">Controls</p>
              <h2>Keyboard and interaction help</h2>
            </div>
            <button type="button" data-close-help>Close</button>
          </div>
          <div class="help-grid">
            <div><strong>1-7</strong><span>Jump to a stage</span></div>
            <div><strong>Arrow keys</strong><span>Previous or next stage</span></div>
            <div><strong>Space</strong><span>Play or pause simulation</span></div>
            <div><strong>R</strong><span>Reset camera</span></div>
            <div><strong>A</strong><span>Toggle auto-rotate</span></div>
            <div><strong>H</strong><span>Toggle this help overlay</span></div>
            <div><strong>F</strong><span>Toggle fullscreen</span></div>
            <div><strong>Esc</strong><span>Close overlays</span></div>
            <div><strong>Mouse drag</strong><span>Orbit and inspect the scene</span></div>
            <div><strong>Wheel / trackpad</strong><span>Zoom in or out</span></div>
            <div><strong>Right drag</strong><span>Pan across the scene</span></div>
          </div>
        </div>
      </div>
    </div>
  `
}

function collectReferences(root: HTMLElement): References {
  const stageButtonContainer = root.querySelector<HTMLElement>('[data-stage-buttons]')
  if (!stageButtonContainer) {
    throw new Error('Failed to locate stage button container.')
  }

  stageButtonContainer.innerHTML = STAGES.map(
    (stage) =>
      `<button type="button" class="stage-button" data-stage-id="${stage.id}"><strong>${stage.label}</strong><span>${stage.title}</span></button>`,
  ).join('')

  return {
    stageButtons: Array.from(stageButtonContainer.querySelectorAll<HTMLButtonElement>('[data-stage-id]')),
    stageLabel: root.querySelector<HTMLElement>('[data-stage-label]')!,
    stageTitle: root.querySelector<HTMLElement>('[data-stage-title]')!,
    stageDescription: root.querySelector<HTMLElement>('[data-stage-description]')!,
    stageFootnote: root.querySelector<HTMLElement>('[data-stage-footnote]')!,
    stageProgress: root.querySelector<HTMLElement>('[data-stage-progress]')!,
    methodStep: root.querySelector<HTMLElement>('[data-method-step]')!,
    methodGrid: root.querySelector<HTMLElement>('[data-method-grid]')!,
    stageOverlayLabel: root.querySelector<HTMLElement>('[data-stage-overlay-label]')!,
    stageOverlayTitle: root.querySelector<HTMLElement>('[data-stage-overlay-title]')!,
    stageOverlayDescription: root.querySelector<HTMLElement>('[data-stage-overlay-description]')!,
    stageOverlayProgress: root.querySelector<HTMLElement>('[data-stage-overlay-progress]')!,
    metrics: root.querySelector<HTMLElement>('[data-metrics]')!,
    routeSelect: root.querySelector<HTMLSelectElement>('[data-route-select]')!,
    scenarioSelect: root.querySelector<HTMLSelectElement>('[data-scenario-select]')!,
    sceneHost: root.querySelector<HTMLElement>('[data-scene-host]')!,
    loading: root.querySelector<HTMLElement>('[data-loading]')!,
    error: root.querySelector<HTMLElement>('[data-error]')!,
    warningPanel: root.querySelector<HTMLElement>('[data-warning-panel]')!,
    warnings: root.querySelector<HTMLElement>('[data-warnings]')!,
    helpOverlay: root.querySelector<HTMLElement>('[data-help-overlay]')!,
    legendPanel: root.querySelector<HTMLElement>('[data-legend]')!,
    pointCloudNote: root.querySelector<HTMLElement>('[data-point-cloud-note]')!,
    prevButton: root.querySelector<HTMLButtonElement>('[data-prev]')!,
    nextButton: root.querySelector<HTMLButtonElement>('[data-next]')!,
    resetButton: root.querySelector<HTMLButtonElement>('[data-reset]')!,
    autoRotateButton: root.querySelector<HTMLButtonElement>('[data-autorotate]')!,
    playButton: root.querySelector<HTMLButtonElement>('[data-play]')!,
    helpButton: root.querySelector<HTMLButtonElement>('[data-help]')!,
    fullscreenButton: root.querySelector<HTMLButtonElement>('[data-fullscreen]')!,
    routeWrap: root.querySelector<HTMLElement>('[data-route-wrap]')!,
    scenarioWrap: root.querySelector<HTMLElement>('[data-scenario-wrap]')!,
    footerNote: root.querySelector<HTMLElement>('[data-footer-note]')!,
  }
}

function summarizeSpecialGraphNodes(graph: NavigationGraphData): string {
  let fareGates = 0
  let entrances = 0
  let verticalLinks = 0

  for (const node of graph.nodes) {
    const normalized = node.nodeType.toLowerCase()
    if (normalized.includes('fare_gate')) {
      fareGates += 1
      continue
    }
    if (normalized.includes('entrance') || normalized.includes('exit')) {
      entrances += 1
      continue
    }
    if (normalized.includes('stair') || normalized.includes('escalator') || normalized.includes('elevator')) {
      verticalLinks += 1
    }
  }

  return [`Gates ${fareGates}`, `Entrances ${entrances}`, `Vertical ${verticalLinks}`].join(' | ')
}

function applyTheme(config: DemoConfig): void {
  const root = document.documentElement
  root.style.setProperty('--accent', config.visuals.accent)
  root.style.setProperty('--accent-soft', config.visuals.accentSoft)
  root.style.setProperty('--bg-top', config.visuals.backgroundTop)
  root.style.setProperty('--bg-bottom', config.visuals.backgroundBottom)
  root.style.setProperty('--graph-color', config.visuals.graph)
  root.style.setProperty('--route-color', config.visuals.route)
  root.style.setProperty('--point-color', config.visuals.pointCloud)
  root.style.setProperty('--movement-color', config.visuals.floor)
  root.style.setProperty('--sim-color', config.visuals.simulation)
}

function formatMetric(value: MetricValue, missingLabel: string): string {
  if (value === null || value === undefined || value === '') {
    return missingLabel
  }

  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  }

  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }

  if (Array.isArray(value)) {
    return value.length > 0 ? value.join(', ') : missingLabel
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value)
    if (entries.length === 0) {
      return missingLabel
    }
    return entries
      .slice(0, 4)
      .map(([key, entry]) => `${key}: ${entry}`)
      .join(' | ')
  }

  return value
}

function metricCard(label: string, value: string): string {
  return `<article class="metric-card"><span>${label}</span><strong>${value}</strong></article>`
}

function methodCard(label: string, value: string): string {
  return `<article class="method-card"><span>${label}</span><strong>${value}</strong></article>`
}

function getStageMetrics(
  stage: StageDefinition,
  config: DemoConfig,
  graph: NavigationGraphData,
  status: SceneStatus,
  route: DemoRoute | undefined,
  scenario: SimulationScenario | undefined,
  totalAnchors: number,
  graphMetrics: Record<string, MetricValue>,
): Array<{ label: string; value: MetricValue }> {
  switch (stage.id) {
    case 1:
      return [
        { label: 'Model mode', value: status.modelMode === 'asset' ? 'Exploded GLB storeys' : 'Exploded storey proxy' },
        { label: 'Presentation', value: 'Whole-station exploded by floor' },
        { label: 'Configured model path', value: config.assets.bimModel.path },
        { label: 'Graph nodes available', value: graphMetrics.total_nodes },
        { label: 'Graph levels', value: graphMetrics.level_node_counts },
      ]
    case 2:
      return [
        { label: 'Pipeline step', value: 'STEP 1 Geometry' },
        { label: 'Geometry source', value: status.movementMode === 'asset' ? 'Extracted walkable surfaces' : 'Walkable shell fallback' },
        { label: 'Connector summary', value: graphMetrics.connectors },
        { label: 'Levels present', value: graphMetrics.level_node_counts },
      ]
    case 3:
      return [
        { label: 'Pipeline step', value: 'STEP 2 Sampling' },
        { label: 'Sampling mode', value: status.pointCloudMode === 'asset' ? 'PLY sample cloud' : 'Aggregated density tiles' },
        { label: 'Nominal grid spacing', value: '0.5 m' },
        { label: 'Valid navigation nodes', value: graphMetrics.total_nodes },
      ]
    case 4:
      return [
        { label: 'Total nodes', value: graphMetrics.total_nodes },
        { label: 'Total edges', value: graphMetrics.total_edges },
        { label: 'Connected graph', value: graphMetrics.is_connected },
        { label: 'Special nodes', value: summarizeSpecialGraphNodes(graph) },
        { label: 'Connector summary', value: graphMetrics.connectors },
      ]
    case 5:
      return [
        { label: 'Route label', value: route?.label ?? null },
        { label: 'Route source', value: route?.sourceKind ?? null },
        { label: 'Travel time', value: route?.metrics.totalTravelTime ?? route?.metrics.total_travel_time ?? null },
        { label: 'Connectors used', value: route?.metrics.connectorsUsed ?? route?.metrics.connectors_used ?? null },
      ]
    case 6:
      return [
        { label: 'Scenario label', value: scenario?.label ?? null },
        { label: 'Scenario kind', value: scenario?.kind ?? null },
        { label: 'Mean travel time', value: scenario?.summary.mean_travel_time ?? null },
        { label: 'Mean wait time', value: scenario?.summary.mean_wait_time ?? null },
        { label: 'Total replans', value: scenario?.summary.total_replans ?? null },
        { label: 'Max queue', value: scenario?.summary.max_queue ?? null },
      ]
    case 7:
      return [
        { label: 'Current route', value: status.routeLabel },
        { label: 'Current simulation', value: status.simulationLabel },
        { label: 'Integrated graph edges', value: graphMetrics.total_edges },
        { label: 'Special nodes', value: summarizeSpecialGraphNodes(graph) },
        { label: 'Anchor count', value: totalAnchors },
      ]
    default:
      return []
  }
}

export async function bootstrapDemo(root: HTMLDivElement): Promise<void> {
  root.innerHTML = buildShell()
  const refs = collectReferences(root)
  const warningMessages: string[] = []

  const pushWarning = (message: string): void => {
    if (warningMessages.includes(message)) {
      return
    }

    warningMessages.unshift(message)
    refs.warningPanel.classList.toggle('hidden', warningMessages.length === 0)
    refs.warnings.innerHTML = warningMessages
      .slice(0, 6)
      .map((item) => `<article class="warning-item">${item}</article>`)
      .join('')
  }

  let scene: DemoScene | null = null

  try {
    const bundle = await loadDemoBundle(pushWarning)
    applyTheme(bundle.config)
    refs.footerNote.textContent = bundle.config.app.footerNote
    refs.routeSelect.innerHTML = bundle.routes.routes
      .map((route) => `<option value="${route.id}">${route.label}</option>`)
      .join('')
    refs.scenarioSelect.innerHTML = bundle.simulation.scenarios
      .map((scenario) => `<option value="${scenario.id}">${scenario.label}</option>`)
      .join('')

    scene = new DemoScene(refs.sceneHost, bundle.config)
    await scene.hydrate(bundle, pushWarning)
    refs.loading.classList.add('hidden')

    let currentStageIndex = 0
    let currentRoute = bundle.routes.routes.find((route) => route.id === bundle.routes.defaultRouteId) ?? bundle.routes.routes[0]
    let currentScenario =
      bundle.simulation.scenarios.find((scenario) => scenario.id === bundle.simulation.defaultScenarioId) ??
      bundle.simulation.scenarios[0]
    let helpVisible = false
    let autoRotate = false

    const render = (): void => {
      const stage = STAGES[currentStageIndex]
      const method = STAGE_METHODS[stage.id]
      const status = scene?.getStatus() ?? {
        routeLabel: currentRoute?.label ?? 'No route loaded',
        simulationLabel: currentScenario?.label ?? 'No simulation loaded',
        modelMode: 'placeholder',
        pointCloudMode: 'placeholder',
        movementMode: 'graph-fallback',
        simulationMode: 'illustrative',
      }

      refs.stageLabel.textContent = `${stage.label} • ${method.step}`
      refs.stageTitle.textContent = stage.title
      refs.stageDescription.textContent = stage.description
      refs.stageFootnote.textContent = stage.footnote ?? ''
      refs.stageProgress.textContent = `${currentStageIndex + 1} / ${STAGES.length}`
      refs.methodStep.textContent = method.step
      refs.methodGrid.innerHTML = [
        methodCard('Input', method.input),
        methodCard('Operation', method.operation),
        methodCard('Output', method.output),
      ].join('')
      refs.stageOverlayLabel.textContent = `${stage.label} • ${method.step}`
      refs.stageOverlayTitle.textContent = stage.title
      refs.stageOverlayDescription.textContent = stage.description
      refs.stageOverlayProgress.textContent = `${currentStageIndex + 1} / ${STAGES.length}`
      refs.pointCloudNote.classList.toggle('hidden', !(stage.id === 3 || stage.id === 7))
      refs.legendPanel.classList.toggle('hidden', stage.id < 4)
      refs.routeWrap.classList.toggle('hidden', !(stage.id === 5 || stage.id === 7 || stage.id === 6))
      refs.scenarioWrap.classList.toggle('hidden', !(stage.id === 6 || stage.id === 7))
      refs.metrics.innerHTML = getStageMetrics(
        stage,
        bundle.config,
        bundle.graph,
        status,
        currentRoute,
        currentScenario,
        bundle.anchors.anchors.length,
        bundle.graph.metrics,
      )
        .map((metric) => metricCard(metric.label, formatMetric(metric.value, bundle.config.metrics.missingLabel)))
        .join('')

      refs.stageButtons.forEach((button, index) => {
        button.classList.toggle('active', index === currentStageIndex)
      })
      refs.autoRotateButton.textContent = autoRotate ? 'Auto-rotate on' : 'Auto-rotate'
      refs.playButton.textContent = 'Play simulation'
    }

    const selectStage = (index: number): void => {
      currentStageIndex = (index + STAGES.length) % STAGES.length
      scene?.setStage(STAGES[currentStageIndex].id)
      render()
    }

    const selectRoute = (routeId: string): void => {
      const nextRoute = bundle.routes.routes.find((route) => route.id === routeId)
      if (!nextRoute) {
        return
      }
      currentRoute = nextRoute
      refs.routeSelect.value = nextRoute.id
      scene?.setRoute(nextRoute)
      render()
    }

    const selectScenario = (scenarioId: string): void => {
      const nextScenario = bundle.simulation.scenarios.find((scenario) => scenario.id === scenarioId)
      if (!nextScenario) {
        return
      }
      currentScenario = nextScenario
      refs.scenarioSelect.value = nextScenario.id
      scene?.setScenario(nextScenario)
      render()
    }

    const toggleHelp = (visible?: boolean): void => {
      helpVisible = visible ?? !helpVisible
      refs.helpOverlay.classList.toggle('hidden', !helpVisible)
    }

    const toggleFullscreen = async (): Promise<void> => {
      if (document.fullscreenElement) {
        await document.exitFullscreen()
      } else {
        await refs.sceneHost.requestFullscreen()
      }
    }

    refs.stageButtons.forEach((button, index) => {
      button.addEventListener('click', () => selectStage(index))
    })
    refs.prevButton.addEventListener('click', () => selectStage(currentStageIndex - 1))
    refs.nextButton.addEventListener('click', () => selectStage(currentStageIndex + 1))
    refs.resetButton.addEventListener('click', () => scene?.resetCamera())
    refs.autoRotateButton.addEventListener('click', () => {
      autoRotate = scene?.toggleAutoRotate() ?? false
      render()
    })
    refs.playButton.addEventListener('click', () => {
      const playing = scene?.toggleSimulationPlayback()
      refs.playButton.textContent = playing ? 'Pause simulation' : 'Play simulation'
    })
    refs.helpButton.addEventListener('click', () => toggleHelp())
    refs.helpOverlay.querySelector<HTMLButtonElement>('[data-close-help]')?.addEventListener('click', () => {
      toggleHelp(false)
    })
    refs.fullscreenButton.addEventListener('click', () => {
      void toggleFullscreen()
    })
    refs.routeSelect.addEventListener('change', (event) => {
      selectRoute((event.target as HTMLSelectElement).value)
    })
    refs.scenarioSelect.addEventListener('change', (event) => {
      selectScenario((event.target as HTMLSelectElement).value)
    })

    window.addEventListener('keydown', (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return
      }

      if (/^[1-7]$/.test(event.key)) {
        selectStage(Number.parseInt(event.key, 10) - 1)
        return
      }

      switch (event.key) {
        case 'ArrowRight':
          event.preventDefault()
          selectStage(currentStageIndex + 1)
          break
        case 'ArrowLeft':
          event.preventDefault()
          selectStage(currentStageIndex - 1)
          break
        case ' ':
          event.preventDefault()
          refs.playButton.click()
          break
        case 'r':
        case 'R':
          scene?.resetCamera()
          break
        case 'a':
        case 'A':
          refs.autoRotateButton.click()
          break
        case 'h':
        case 'H':
          toggleHelp()
          break
        case 'f':
        case 'F':
          void toggleFullscreen()
          break
        case 'Escape':
          toggleHelp(false)
          break
      }
    })

    refs.routeSelect.value = currentRoute?.id ?? ''
    refs.scenarioSelect.value = currentScenario?.id ?? ''
    render()
  } catch (error) {
    refs.loading.classList.add('hidden')
    refs.error.classList.remove('hidden')
    refs.error.innerHTML = `
      <div class="error-card">
        <p class="eyebrow">Load failure</p>
        <h2>Unable to initialize the demonstrator</h2>
        <p>${error instanceof Error ? error.message : String(error)}</p>
      </div>
    `
  }

  window.addEventListener('beforeunload', () => {
    scene?.destroy()
  })
}
