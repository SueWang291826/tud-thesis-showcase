import { ensureRouteGeometry } from './dijkstra'
import { resolvePublicAssetUrl } from './paths'
import type {
  AssetReference,
  DemoConfig,
  DemoDataBundle,
  DemoRoute,
  GraphMeta,
  MetricValue,
  MovementGeometryData,
  MovementSurface,
  NavigationEdge,
  NavigationGraphData,
  NavigationNode,
  RoutesData,
  SemanticAnchorData,
  SimulationData,
  SimulationFrame,
  SimulationScenario,
  Vec3,
} from './types'

const EXACT_POINT_CLOUD_DISCLAIMER =
  'This point cloud is a BIM-derived, surface-sampled visualization generated from IFC/mesh geometry. It is not measured LiDAR, photogrammetry, or any other sensor-captured reality data.'

const DEFAULT_CONFIG: DemoConfig = {
  app: {
    title: 'Interactive BIM-to-Navigation Digital Twin Demonstrator',
    subtitle:
      'A local thesis-defense walkthrough from BIM geometry to graph, routing, and simulation.',
    footerNote:
      'Runs locally in the browser with configurable placeholders for missing BIM and point-cloud assets.',
    pointCloudDisclaimer: EXACT_POINT_CLOUD_DISCLAIMER,
    gestureNotice:
      'Gesture control is optional. Keyboard and mouse controls remain active even when MediaPipe assets or webcam access are unavailable.',
  },
  assets: {
    bimModel: {
      path: '/models/station_bim.glb',
      placeholderAllowed: true,
      transform: {
        offset: { x: 0, y: 0, z: 0 },
        rotationDeg: { x: 0, y: 0, z: 0 },
        scale: 1,
      },
    },
    pointCloud: {
      path: '/pointcloud/station_surface_sampled.ply',
      placeholderAllowed: true,
      transform: {
        offset: { x: 0, y: 0, z: 0 },
        rotationDeg: { x: 0, y: 0, z: 0 },
        scale: 1,
      },
    },
    movementGeometry: {
      path: '/data/movement_geometry.json',
      optional: true,
      placeholderAllowed: true,
    },
    navigationGraph: {
      path: '/data/navigation_graph.json',
    },
    semanticAnchors: {
      path: '/data/semantic_anchors.json',
    },
    routes: {
      path: '/data/routes.json',
    },
    simulation: {
      path: '/data/simulation.json',
      optional: true,
      placeholderAllowed: true,
    },
    mediapipe: {
      enabled: true,
      wasmPath: '/mediapipe/wasm',
      modelAssetPath: '/mediapipe/gesture_recognizer.task',
      minScore: 0.6,
      cooldownMs: 1200,
    },
  },
  visuals: {
    backgroundTop: '#14324a',
    backgroundBottom: '#03080e',
    accent: '#ff9f1c',
    accentSoft: '#2ec4b6',
    route: '#ffd166',
    graph: '#4cc9f0',
    floor: '#f4a261',
    anchor: '#ff595e',
    simulation: '#80ed99',
    pointCloud: '#f7e1a0',
    transitionMs: 720,
    autoRotateSpeed: 0.35,
    pointSize: 0.22,
    graphRenderStep: 2,
    movementRenderStep: 6,
    simulationPointSize: 0.42,
  },
  camera: {
    position: { x: 110, y: 74, z: -120 },
    target: { x: 70, y: 8, z: -15 },
    fov: 50,
    near: 0.1,
    far: 2000,
  },
  controls: {
    enableDamping: true,
    dampingFactor: 0.08,
    minDistance: 15,
    maxDistance: 380,
  },
  metrics: {
    missingLabel: 'Not provided',
  },
  placeholderFlags: {
    model: true,
    pointCloud: true,
    movementGeometry: true,
    simulation: false,
  },
}

function isoNow(): string {
  return new Date().toISOString()
}

function createMeta(sourcePaths: string[]): GraphMeta {
  return {
    generatedAt: isoNow(),
    generator: 'src/data-loader.ts fallback',
    sourcePaths,
  }
}

function sortLevels(levels: Iterable<string>): string[] {
  return Array.from(new Set(levels)).sort((left, right) => {
    const extract = (value: string): number => {
      const match = value.match(/F(\d+)/i)
      return match ? Number.parseInt(match[1], 10) : Number.POSITIVE_INFINITY
    }

    const leftFloor = extract(left)
    const rightFloor = extract(right)
    if (Number.isFinite(leftFloor) || Number.isFinite(rightFloor)) {
      if (leftFloor !== rightFloor) {
        return leftFloor - rightFloor
      }
    }

    return left.localeCompare(right)
  })
}

function computeBounds(nodes: NavigationNode[]): NavigationGraphData['bounds'] {
  const initial = {
    min: { x: Number.POSITIVE_INFINITY, y: Number.POSITIVE_INFINITY, z: Number.POSITIVE_INFINITY },
    max: { x: Number.NEGATIVE_INFINITY, y: Number.NEGATIVE_INFINITY, z: Number.NEGATIVE_INFINITY },
  }

  return nodes.reduce((bounds, node) => ({
    min: {
      x: Math.min(bounds.min.x, node.x),
      y: Math.min(bounds.min.y, node.y),
      z: Math.min(bounds.min.z, node.z),
    },
    max: {
      x: Math.max(bounds.max.x, node.x),
      y: Math.max(bounds.max.y, node.y),
      z: Math.max(bounds.max.z, node.z),
    },
  }), initial)
}

function fallbackNodes(): NavigationNode[] {
  return [
    { id: 'F1_platform_west', x: 10, y: 6, z: 0, level: 'F1', nodeType: 'platform' },
    { id: 'F1_platform_mid', x: 28, y: 6, z: 0, level: 'F1', nodeType: 'floor' },
    { id: 'F1_platform_east', x: 46, y: 6, z: 0, level: 'F1', nodeType: 'platform' },
    { id: 'F1_stair_entry', x: 28, y: 14, z: 0, level: 'F1', nodeType: 'stair_chain' },
    { id: 'F3_stair_landing', x: 28, y: 14, z: 6, level: 'F3', nodeType: 'stair_chain' },
    { id: 'F3_fare_gate', x: 28, y: 22, z: 6, level: 'F3', nodeType: 'fare_gate_entry' },
    { id: 'F3_concourse_west', x: 14, y: 30, z: 6, level: 'F3', nodeType: 'floor' },
    { id: 'F3_concourse_east', x: 42, y: 30, z: 6, level: 'F3', nodeType: 'floor' },
    { id: 'F3_upper_stair', x: 28, y: 38, z: 6, level: 'F3', nodeType: 'stair_chain' },
    { id: 'F4_landing', x: 28, y: 38, z: 12, level: 'F4', nodeType: 'stair_chain' },
    { id: 'F4_entrance_left', x: 16, y: 46, z: 12, level: 'F4', nodeType: 'entrance' },
    { id: 'F4_entrance_right', x: 40, y: 46, z: 12, level: 'F4', nodeType: 'entrance' },
  ]
}

function fallbackEdges(): NavigationEdge[] {
  const baseEdges: NavigationEdge[] = [
    {
      id: 'e1',
      source: 'F1_platform_west',
      target: 'F1_platform_mid',
      length2d: 18,
      length3d: 18,
      travelTime: 15,
      edgeType: 'floor',
      level: 'F1',
    },
    {
      id: 'e2',
      source: 'F1_platform_mid',
      target: 'F1_platform_east',
      length2d: 18,
      length3d: 18,
      travelTime: 15,
      edgeType: 'floor',
      level: 'F1',
    },
    {
      id: 'e3',
      source: 'F1_platform_mid',
      target: 'F1_stair_entry',
      length2d: 8,
      length3d: 8,
      travelTime: 6,
      edgeType: 'floor',
      level: 'F1',
    },
    {
      id: 'e4',
      source: 'F1_stair_entry',
      target: 'F3_stair_landing',
      length2d: 0,
      length3d: 6,
      travelTime: 13,
      edgeType: 'stair',
      level: 'STAIR',
    },
    {
      id: 'e5',
      source: 'F3_stair_landing',
      target: 'F3_fare_gate',
      length2d: 8,
      length3d: 8,
      travelTime: 6,
      edgeType: 'floor',
      level: 'F3',
    },
    {
      id: 'e6',
      source: 'F3_fare_gate',
      target: 'F3_concourse_west',
      length2d: 16,
      length3d: 16,
      travelTime: 13,
      edgeType: 'fare_gate',
      level: 'F3',
    },
    {
      id: 'e7',
      source: 'F3_fare_gate',
      target: 'F3_concourse_east',
      length2d: 16,
      length3d: 16,
      travelTime: 13,
      edgeType: 'fare_gate',
      level: 'F3',
    },
    {
      id: 'e8',
      source: 'F3_fare_gate',
      target: 'F3_upper_stair',
      length2d: 16,
      length3d: 16,
      travelTime: 13,
      edgeType: 'floor',
      level: 'F3',
    },
    {
      id: 'e9',
      source: 'F3_upper_stair',
      target: 'F4_landing',
      length2d: 0,
      length3d: 6,
      travelTime: 13,
      edgeType: 'stair',
      level: 'STAIR',
    },
    {
      id: 'e10',
      source: 'F4_landing',
      target: 'F4_entrance_left',
      length2d: 14,
      length3d: 14,
      travelTime: 12,
      edgeType: 'entrance',
      level: 'F4',
    },
    {
      id: 'e11',
      source: 'F4_landing',
      target: 'F4_entrance_right',
      length2d: 14,
      length3d: 14,
      travelTime: 12,
      edgeType: 'entrance',
      level: 'F4',
    },
  ]

  return baseEdges
}

function fallbackGraph(): NavigationGraphData {
  const nodes = fallbackNodes()
  const edges = fallbackEdges()
  const levelCounts = nodes.reduce<Record<string, number>>((accumulator, node) => {
    accumulator[node.level] = (accumulator[node.level] ?? 0) + 1
    return accumulator
  }, {})
  const edgeTypeCounts = edges.reduce<Record<string, number>>((accumulator, edge) => {
    accumulator[edge.edgeType] = (accumulator[edge.edgeType] ?? 0) + 1
    return accumulator
  }, {})

  return {
    meta: createMeta(['fallback://graph']),
    nodes,
    edges,
    levels: sortLevels(nodes.map((node) => node.level)),
    bounds: computeBounds(nodes),
    metrics: {
      total_nodes: nodes.length,
      total_edges: edges.length,
      level_node_counts: levelCounts,
      edge_types: edgeTypeCounts,
      is_connected: true,
      note: 'Fallback graph used because prepared thesis graph data is unavailable.',
    },
  }
}

function fallbackAnchors(): SemanticAnchorData {
  return {
    meta: createMeta(['fallback://anchors']),
    anchors: [
      { id: 'anchor_platform', type: 'PLATFORM', level: 'F1', x: 10, y: 6, z: 0 },
      { id: 'anchor_gate', type: 'FARE_GATE', level: 'F3', x: 28, y: 22, z: 6 },
      { id: 'anchor_exit_left', type: 'EXIT', level: 'F4', x: 16, y: 46, z: 12 },
      { id: 'anchor_exit_right', type: 'EXIT', level: 'F4', x: 40, y: 46, z: 12 },
    ],
    counts: {
      PLATFORM: 1,
      FARE_GATE: 1,
      EXIT: 2,
    },
  }
}

function fallbackRoutes(graph: NavigationGraphData): RoutesData {
  const route: DemoRoute = {
    id: 'fallback_route_platform_to_exit',
    label: 'Platform to entrance',
    originNodeId: 'F1_platform_west',
    destinationNodeId: 'F4_entrance_right',
    sourceKind: 'derived-dijkstra',
    metrics: {
      note: 'Fallback route used because prepared thesis route data is unavailable.',
    },
  }

  return {
    meta: createMeta(['fallback://routes']),
    routes: [ensureRouteGeometry(route, graph)],
    defaultRouteId: route.id,
  }
}

function distance3d(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z)
}

function interpolatePolyline(polyline: Vec3[], ratio: number): Vec3 {
  if (polyline.length === 0) {
    return { x: 0, y: 0, z: 0 }
  }

  if (polyline.length === 1) {
    return polyline[0]
  }

  const segmentLengths = polyline.slice(1).map((point, index) => distance3d(point, polyline[index]))
  const totalLength = segmentLengths.reduce((sum, value) => sum + value, 0)
  const targetLength = totalLength * Math.max(0, Math.min(1, ratio))
  let accumulated = 0

  for (let index = 0; index < segmentLengths.length; index += 1) {
    const segmentLength = segmentLengths[index]
    if (accumulated + segmentLength >= targetLength) {
      const localRatio = segmentLength === 0 ? 0 : (targetLength - accumulated) / segmentLength
      const start = polyline[index]
      const end = polyline[index + 1]
      return {
        x: start.x + (end.x - start.x) * localRatio,
        y: start.y + (end.y - start.y) * localRatio,
        z: start.z + (end.z - start.z) * localRatio,
      }
    }

    accumulated += segmentLength
  }

  return polyline[polyline.length - 1]
}

function fallbackSimulation(routes: RoutesData): SimulationData {
  const route = routes.routes[0]
  const polyline = route.polyline ?? []
  const frames: SimulationFrame[] = []
  const frameCount = 36
  const agentCount = 8

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const t = frameIndex * 1.5
    const agents = Array.from({ length: agentCount }, (_, agentIndex) => {
      const ratio = Math.max(0, Math.min(1, (frameIndex - agentIndex * 2) / (frameCount - 1)))
      const position = interpolatePolyline(polyline, ratio)
      return {
        id: `illustrative_${agentIndex + 1}`,
        x: position.x,
        y: position.y,
        z: position.z,
      }
    })

    frames.push({ t, agents })
  }

  const scenario: SimulationScenario = {
    id: 'illustrative_dynamic',
    label: 'Illustrative fallback simulation',
    kind: 'illustrative',
    routingMode: 'illustrative',
    summary: {
      note: 'Illustrative fallback simulation shown because prepared thesis simulation data is unavailable.',
      n_agents: agentCount,
      mean_travel_time: null,
      mean_wait_time: null,
      total_replans: null,
    },
    frames,
    timeline: {
      start: frames[0]?.t ?? 0,
      end: frames[frames.length - 1]?.t ?? 0,
      step: 1.5,
    },
  }

  return {
    meta: createMeta(['fallback://simulation']),
    scenarios: [scenario],
    defaultScenarioId: scenario.id,
  }
}

function deriveMovementGeometry(
  graph: NavigationGraphData,
  renderStep: number,
): MovementGeometryData {
  const byLevel = new Map<string, NavigationNode[]>()

  for (const node of graph.nodes) {
    if (!node.nodeType.includes('floor') && node.nodeType !== 'platform') {
      continue
    }

    const list = byLevel.get(node.level) ?? []
    list.push(node)
    byLevel.set(node.level, list)
  }

  const surfaces: MovementSurface[] = Array.from(byLevel.entries()).map(([level, nodes]) => ({
    id: `movement_${level.toLowerCase()}`,
    level,
    label: `${level} walkable sampling`,
    source: 'graph-fallback',
    points: nodes
      .filter((_, index) => index % Math.max(1, renderStep) === 0)
      .map((node) => ({ x: node.x, y: node.y, z: node.z })),
  }))

  return {
    meta: createMeta(['fallback://movement-from-graph']),
    surfaces,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function coerceMetricValue(value: unknown): MetricValue {
  if (value === null || typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
    return value
  }

  if (Array.isArray(value) && value.every((entry) => typeof entry === 'string')) {
    return value
  }

  if (isRecord(value)) {
    const numericEntries = Object.entries(value).every(([, entry]) => typeof entry === 'number')
    if (numericEntries) {
      return Object.fromEntries(
        Object.entries(value).map(([key, entry]) => [key, Number(entry)]),
      )
    }
  }

  return JSON.stringify(value)
}

function mergeConfig(loaded: Partial<DemoConfig> | null): DemoConfig {
  if (!loaded) {
    return DEFAULT_CONFIG
  }

  return {
    ...DEFAULT_CONFIG,
    ...loaded,
    app: {
      ...DEFAULT_CONFIG.app,
      ...loaded.app,
      pointCloudDisclaimer: EXACT_POINT_CLOUD_DISCLAIMER,
    },
    assets: {
      ...DEFAULT_CONFIG.assets,
      ...loaded.assets,
      bimModel: {
        ...DEFAULT_CONFIG.assets.bimModel,
        ...loaded.assets?.bimModel,
      },
      pointCloud: {
        ...DEFAULT_CONFIG.assets.pointCloud,
        ...loaded.assets?.pointCloud,
      },
      movementGeometry: {
        ...DEFAULT_CONFIG.assets.movementGeometry,
        ...loaded.assets?.movementGeometry,
      },
      navigationGraph: {
        ...DEFAULT_CONFIG.assets.navigationGraph,
        ...loaded.assets?.navigationGraph,
      },
      semanticAnchors: {
        ...DEFAULT_CONFIG.assets.semanticAnchors,
        ...loaded.assets?.semanticAnchors,
      },
      routes: {
        ...DEFAULT_CONFIG.assets.routes,
        ...loaded.assets?.routes,
      },
      simulation: {
        ...DEFAULT_CONFIG.assets.simulation,
        ...loaded.assets?.simulation,
      },
      mediapipe: {
        ...DEFAULT_CONFIG.assets.mediapipe,
        ...loaded.assets?.mediapipe,
      },
    },
    visuals: {
      ...DEFAULT_CONFIG.visuals,
      ...loaded.visuals,
    },
    camera: {
      ...DEFAULT_CONFIG.camera,
      ...loaded.camera,
      position: {
        ...DEFAULT_CONFIG.camera.position,
        ...loaded.camera?.position,
      },
      target: {
        ...DEFAULT_CONFIG.camera.target,
        ...loaded.camera?.target,
      },
    },
    controls: {
      ...DEFAULT_CONFIG.controls,
      ...loaded.controls,
    },
    metrics: {
      ...DEFAULT_CONFIG.metrics,
      ...loaded.metrics,
    },
    placeholderFlags: {
      ...DEFAULT_CONFIG.placeholderFlags,
      ...loaded.placeholderFlags,
    },
  }
}

async function fetchJson<T>(
  reference: AssetReference,
  fallback: T | null,
  label: string,
  onWarning: (message: string) => void,
): Promise<T> {
  try {
    const response = await fetch(resolvePublicAssetUrl(reference.path), {
      cache: 'no-cache',
    })

    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`)
    }

    return (await response.json()) as T
  } catch (error) {
    if (fallback !== null && (reference.optional || reference.placeholderAllowed)) {
      onWarning(`Could not load ${label} from ${reference.path}. Using fallback data instead.`)
      return fallback
    }

    throw new Error(
      `Unable to load ${label} from ${reference.path}: ${error instanceof Error ? error.message : String(error)}`,
    )
  }
}

function normalizeGraph(graph: NavigationGraphData): NavigationGraphData {
  const nodes = graph.nodes.map((node) => ({
    ...node,
    usable: node.usable ?? true,
    clearance: typeof node.clearance === 'number' ? node.clearance : null,
  }))
  const edges = graph.edges.map((edge, index) => ({
    ...edge,
    id: edge.id || `edge_${index}`,
    length2d: Number(edge.length2d),
    length3d: Number(edge.length3d),
    travelTime: Number(edge.travelTime),
  }))

  return {
    ...graph,
    nodes,
    edges,
    levels: graph.levels?.length ? sortLevels(graph.levels) : sortLevels(nodes.map((node) => node.level)),
    bounds: graph.bounds ?? computeBounds(nodes),
    metrics: Object.fromEntries(
      Object.entries(graph.metrics ?? {}).map(([key, value]) => [key, coerceMetricValue(value)]),
    ),
  }
}

function normalizeAnchors(anchors: SemanticAnchorData): SemanticAnchorData {
  const counts = anchors.counts ?? anchors.anchors.reduce<Record<string, number>>((accumulator, anchor) => {
    accumulator[anchor.type] = (accumulator[anchor.type] ?? 0) + 1
    return accumulator
  }, {})

  return {
    ...anchors,
    counts,
  }
}

function normalizeRoutes(routes: RoutesData, graph: NavigationGraphData): RoutesData {
  const normalizedRoutes = routes.routes.map((route) =>
    ensureRouteGeometry(
      {
        ...route,
        metrics: Object.fromEntries(
          Object.entries(route.metrics ?? {}).map(([key, value]) => [key, coerceMetricValue(value)]),
        ),
      },
      graph,
    ),
  )

  return {
    ...routes,
    routes: normalizedRoutes,
    defaultRouteId: routes.defaultRouteId ?? normalizedRoutes[0]?.id,
  }
}

function normalizeSimulation(
  simulation: SimulationData,
  routes: RoutesData,
  onWarning: (message: string) => void,
): SimulationData {
  if (!simulation.scenarios?.length) {
    onWarning('Simulation data is empty. Falling back to an illustrative scenario.')
    return fallbackSimulation(routes)
  }

  const scenarios = simulation.scenarios.map((scenario) => {
    const frames = scenario.frames
      .map((frame) => ({
        t: Number(frame.t),
        agents: frame.agents.map((agent) => ({
          id: agent.id,
          x: Number(agent.x),
          y: Number(agent.y),
          z: Number(agent.z),
        })),
      }))
      .sort((left, right) => left.t - right.t)

    const agentMeta = Object.fromEntries(
      Object.entries(scenario.agentMeta ?? {}).map(([agentId, meta]) => [
        agentId,
        {
          originNodeId: meta.originNodeId,
          destinationNodeId: meta.destinationNodeId,
          agentType: meta.agentType ?? 'normal',
          spawnTime: typeof meta.spawnTime === 'number' ? meta.spawnTime : Number(meta.spawnTime ?? 0),
        },
      ]),
    )

    const replanEvents = (scenario.replanEvents ?? [])
      .map((event) => ({
        t: Number(event.t),
        agentId: event.agentId,
        newPathLength:
          typeof event.newPathLength === 'number'
            ? event.newPathLength
            : event.newPathLength == null
              ? undefined
              : Number(event.newPathLength),
      }))
      .sort((left, right) => left.t - right.t)

    const derivedStep = frames.length > 1 ? frames[1].t - frames[0].t : scenario.timeline?.step ?? 1

    return {
      ...scenario,
      kind: scenario.kind ?? 'loaded',
      frames,
      agentMeta,
      replanEvents,
      timeline: {
        start: frames[0]?.t ?? scenario.timeline?.start ?? 0,
        end: frames[frames.length - 1]?.t ?? scenario.timeline?.end ?? 0,
        step: derivedStep,
      },
    }
  })

  return {
    ...simulation,
    scenarios,
    defaultScenarioId: simulation.defaultScenarioId ?? scenarios[0]?.id,
  }
}

export async function loadDemoBundle(
  onWarning: (message: string) => void,
): Promise<DemoDataBundle> {
  const rawConfig = await fetchJson<Partial<DemoConfig>>(
    { path: '/config/demo-config.json', optional: true, placeholderAllowed: true },
    DEFAULT_CONFIG,
    'demo config',
    onWarning,
  )
  const config = mergeConfig(rawConfig)

  const graph = normalizeGraph(
    await fetchJson<NavigationGraphData>(
      config.assets.navigationGraph,
      fallbackGraph(),
      'navigation graph',
      onWarning,
    ),
  )
  const anchors = normalizeAnchors(
    await fetchJson<SemanticAnchorData>(
      config.assets.semanticAnchors,
      fallbackAnchors(),
      'semantic anchors',
      onWarning,
    ),
  )
  const routes = normalizeRoutes(
    await fetchJson<RoutesData>(
      config.assets.routes,
      fallbackRoutes(graph),
      'routes',
      onWarning,
    ),
    graph,
  )
  const movement = await fetchJson<MovementGeometryData>(
    config.assets.movementGeometry,
    deriveMovementGeometry(graph, config.visuals.movementRenderStep),
    'movement geometry',
    onWarning,
  )
  const normalizedMovement: MovementGeometryData = {
    ...movement,
    surfaces: movement.surfaces.map((surface) => ({
      ...surface,
      source: surface.source ?? 'asset',
      points: surface.points.map((point) => ({
        x: Number(point.x),
        y: Number(point.y),
        z: Number(point.z),
      })),
    })),
  }
  const simulation = normalizeSimulation(
    await fetchJson<SimulationData>(
      config.assets.simulation,
      fallbackSimulation(routes),
      'simulation data',
      onWarning,
    ),
    routes,
    onWarning,
  )

  return {
    config,
    graph,
    anchors,
    routes,
    simulation,
    movement: normalizedMovement,
  }
}

export { EXACT_POINT_CLOUD_DISCLAIMER }
