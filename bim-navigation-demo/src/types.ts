export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface AssetTransform {
  offset?: Vec3
  rotationDeg?: Vec3
  scale?: number
}

export interface AssetReference {
  path: string
  optional?: boolean
  placeholderAllowed?: boolean
  label?: string
  transform?: AssetTransform
}

export interface DemoConfig {
  app: {
    title: string
    subtitle: string
    footerNote: string
    pointCloudDisclaimer: string
    gestureNotice: string
  }
  assets: {
    bimModel: AssetReference
    pointCloud: AssetReference
    movementGeometry: AssetReference
    navigationGraph: AssetReference
    semanticAnchors: AssetReference
    routes: AssetReference
    simulation: AssetReference
    mediapipe: {
      enabled: boolean
      wasmPath: string
      modelAssetPath: string
      minScore: number
      cooldownMs: number
    }
  }
  visuals: {
    backgroundTop: string
    backgroundBottom: string
    accent: string
    accentSoft: string
    route: string
    graph: string
    floor: string
    anchor: string
    simulation: string
    pointCloud: string
    transitionMs: number
    autoRotateSpeed: number
    pointSize: number
    graphRenderStep: number
    movementRenderStep: number
    simulationPointSize: number
  }
  camera: {
    position: Vec3
    target: Vec3
    fov: number
    near: number
    far: number
  }
  controls: {
    enableDamping: boolean
    dampingFactor: number
    minDistance: number
    maxDistance: number
  }
  metrics: {
    missingLabel: string
  }
  placeholderFlags: {
    model: boolean
    pointCloud: boolean
    movementGeometry: boolean
    simulation: boolean
  }
}

export interface GraphMeta {
  generatedAt: string
  generator: string
  sourcePaths: string[]
}

export interface Bounds {
  min: Vec3
  max: Vec3
}

export interface NavigationNode {
  id: string
  x: number
  y: number
  z: number
  level: string
  nodeType: string
  usable?: boolean
  clearance?: number | null
  blindCategory?: string
  surfaceType?: string
}

export interface NavigationEdge {
  id: string
  source: string
  target: string
  length2d: number
  length3d: number
  travelTime: number
  edgeType: string
  level: string
}

export type MetricValue = string | number | boolean | null | string[] | Record<string, number>

export interface NavigationGraphData {
  meta: GraphMeta
  nodes: NavigationNode[]
  edges: NavigationEdge[]
  levels: string[]
  bounds: Bounds
  metrics: Record<string, MetricValue>
}

export interface SemanticAnchor {
  id: string
  type: string
  level: string
  x: number
  y: number
  z: number
}

export interface SemanticAnchorData {
  meta: GraphMeta
  anchors: SemanticAnchor[]
  counts: Record<string, number>
}

export interface DemoRoute {
  id: string
  label: string
  originNodeId: string
  destinationNodeId: string
  sourceKind: 'provided' | 'derived-dijkstra'
  snappedNodeIds?: string[]
  polyline?: Vec3[]
  metrics: Record<string, MetricValue>
}

export interface RoutesData {
  meta: GraphMeta
  routes: DemoRoute[]
  defaultRouteId?: string
}

export interface SimulationSummary {
  [key: string]: string | number | null
}

export interface SimulationAgentPose {
  id: string
  x: number
  y: number
  z: number
}

export interface SimulationFrame {
  t: number
  agents: SimulationAgentPose[]
}

export interface SimulationScenario {
  id: string
  label: string
  kind: 'loaded' | 'illustrative'
  routingMode: string
  summary: SimulationSummary
  frames: SimulationFrame[]
  timeline: {
    start: number
    end: number
    step: number
  }
}

export interface SimulationData {
  meta: GraphMeta
  scenarios: SimulationScenario[]
  defaultScenarioId?: string
}

export interface MovementSurface {
  id: string
  level: string
  label: string
  source: 'asset' | 'graph-fallback'
  points: Vec3[]
}

export interface MovementGeometryData {
  meta: GraphMeta
  surfaces: MovementSurface[]
}

export interface DemoDataBundle {
  config: DemoConfig
  graph: NavigationGraphData
  anchors: SemanticAnchorData
  routes: RoutesData
  simulation: SimulationData
  movement: MovementGeometryData
}

export interface StageDefinition {
  id: 1 | 2 | 3 | 4 | 5 | 6
  label: string
  title: string
  description: string
  footnote?: string
}

export interface ComputedRoute {
  nodeIds: string[]
  polyline: Vec3[]
  totalTravelTime: number
  totalLength2d: number
  totalLength3d: number
  edgeTypes: Record<string, number>
  levelsVisited: string[]
}

export interface SceneStatus {
  routeLabel: string
  simulationLabel: string
  modelMode: 'asset' | 'placeholder'
  pointCloudMode: 'asset' | 'placeholder'
  movementMode: 'asset' | 'graph-fallback'
  simulationMode: 'loaded' | 'illustrative'
}

export type GestureCommand = 'next' | 'prev' | 'toggleSimulation' | 'toggleAutoRotate' | 'zoomIn' | 'zoomOut'

export type GestureManipulationSource = 'mediapipe' | 'motion-fallback' | 'pointer-fallback'

export interface GestureManipulationEvent {
  phase: 'start' | 'move' | 'end'
  source: GestureManipulationSource
  deltaX: number
  deltaY: number
}
