import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { resolvePublicAssetUrl } from './paths'
import type {
  AssetTransform,
  DemoConfig,
  DemoDataBundle,
  DemoRoute,
  MovementGeometryData,
  NavigationGraphData,
  NavigationNode,
  SceneStatus,
  SimulationScenario,
  StageDefinition,
  Vec3,
} from './types'

interface CameraTween {
  startPosition: THREE.Vector3
  endPosition: THREE.Vector3
  startTarget: THREE.Vector3
  endTarget: THREE.Vector3
  startedAt: number
  duration: number
}

interface FootprintPoint {
  x: number
  y: number
}

interface StoreyDescriptor {
  level: string
  z: number
  order: number
  footprint: FootprintPoint[]
}

type SpecialNodeKind = 'fare_gate' | 'entrance' | 'stair' | 'escalator' | 'elevator'

interface TileFieldSample {
  x: number
  y: number
  z: number
  level: string
}

interface TileFieldOptions {
  cellSize: number
  minHeight: number
  heightStep: number
  maxHeight: number
  opacity: number
  renderOrder: number
  palette: 'pointCloud' | 'movement'
}

interface ScenarioDensityCell {
  sumX: number
  sumY: number
  sumZ: number
  samples: number
  density: number
}

interface SimulationPointSet {
  geometry: THREE.BufferGeometry
  positions: Float32Array
  glow: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>
  core: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>
}

const SIMULATION_SCENARIO_COLORS = {
  static: {
    normal: new THREE.Color('#34f5ff'),
    elderly: new THREE.Color('#c8ff32'),
  },
  dynamic: {
    normal: new THREE.Color('#ff9a1f'),
    elderly: new THREE.Color('#ff4df8'),
  },
} as const

const SIMULATION_POINT_SCALE = 2.8
const SIMULATION_ELDERLY_SCALE = 3.8
const SIMULATION_GLOW_SCALE = 2.1
const REPLAN_MARKER_LIFETIME_FRAMES = 8
const REPLAN_MARKER_HEIGHT = 4.1
const REPLAN_PREVIEW_LEAD_SECONDS = 3

function isStaticScenario(scenario: SimulationScenario): boolean {
  const signature = `${scenario.id} ${scenario.label} ${scenario.routingMode}`.toLowerCase()
  return signature.includes('static')
}

function normalizeSimulationAgentType(agentType?: string): 'normal' | 'elderly' {
  return typeof agentType === 'string' && agentType.toLowerCase().includes('elder') ? 'elderly' : 'normal'
}

function sampleSimulationAgentColor(scenario: SimulationScenario, agentType?: string): THREE.Color {
  const family = isStaticScenario(scenario) ? SIMULATION_SCENARIO_COLORS.static : SIMULATION_SCENARIO_COLORS.dynamic
  return family[normalizeSimulationAgentType(agentType)]
}

function createReplanMarkerTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 180
  canvas.height = 180
  const context = canvas.getContext('2d')
  if (!context) {
    const texture = new THREE.CanvasTexture(canvas)
    texture.needsUpdate = true
    return texture
  }

  context.clearRect(0, 0, canvas.width, canvas.height)
  context.shadowColor = 'rgba(255, 94, 32, 0.55)'
  context.shadowBlur = 18
  context.fillStyle = '#ffe66d'
  context.beginPath()
  context.roundRect(42, 24, 96, 104, 30)
  context.fill()
  context.beginPath()
  context.moveTo(90, 126)
  context.lineTo(76, 154)
  context.lineTo(106, 132)
  context.closePath()
  context.fill()
  context.shadowBlur = 0
  context.lineWidth = 10
  context.strokeStyle = '#ff8c42'
  context.beginPath()
  context.roundRect(42, 24, 96, 104, 30)
  context.stroke()
  context.beginPath()
  context.moveTo(90, 126)
  context.lineTo(76, 154)
  context.lineTo(106, 132)
  context.closePath()
  context.stroke()
  context.fillStyle = '#0b1420'
  context.font = '900 100px Segoe UI, sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText('!', 90, 74)

  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

function isPrimaryStorey(level: string): boolean {
  return /^F\d+/i.test(level)
}

function isWalkableStoreyNode(node: { level: string; nodeType: string; usable?: boolean }): boolean {
  if (!isPrimaryStorey(node.level) || node.usable === false) {
    return false
  }

  return !['stair', 'escalator', 'elevator'].some((token) => node.nodeType.includes(token))
}

function uniqueFootprintPoints(points: FootprintPoint[]): FootprintPoint[] {
  const seen = new Set<string>()
  const unique: FootprintPoint[] = []

  for (const point of points) {
    const key = `${point.x.toFixed(3)}:${point.y.toFixed(3)}`
    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    unique.push(point)
  }

  return unique
}

function cross(origin: FootprintPoint, left: FootprintPoint, right: FootprintPoint): number {
  return (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x)
}

function buildConvexHull(points: FootprintPoint[]): FootprintPoint[] {
  const unique = uniqueFootprintPoints(points).sort((left, right) => {
    if (left.x !== right.x) {
      return left.x - right.x
    }

    return left.y - right.y
  })

  if (unique.length <= 3) {
    return unique
  }

  const lower: FootprintPoint[] = []
  for (const point of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop()
    }
    lower.push(point)
  }

  const upper: FootprintPoint[] = []
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop()
    }
    upper.push(point)
  }

  lower.pop()
  upper.pop()
  return [...lower, ...upper]
}

function buildBoundingRectangle(points: FootprintPoint[]): FootprintPoint[] {
  const unique = uniqueFootprintPoints(points)
  if (unique.length === 0) {
    return []
  }

  const bounds = unique.reduce(
    (result, point) => ({
      minX: Math.min(result.minX, point.x),
      maxX: Math.max(result.maxX, point.x),
      minY: Math.min(result.minY, point.y),
      maxY: Math.max(result.maxY, point.y),
    }),
    {
      minX: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      minY: Number.POSITIVE_INFINITY,
      maxY: Number.NEGATIVE_INFINITY,
    },
  )

  return [
    { x: bounds.minX, y: bounds.minY },
    { x: bounds.maxX, y: bounds.minY },
    { x: bounds.maxX, y: bounds.maxY },
    { x: bounds.minX, y: bounds.maxY },
  ]
}

function toSceneVector(point: Vec3): THREE.Vector3 {
  return new THREE.Vector3(point.x, point.z, -point.y)
}

function degToRad(value = 0): number {
  return THREE.MathUtils.degToRad(value)
}

function applyTransform(target: THREE.Object3D, transform?: AssetTransform): void {
  if (!transform) {
    return
  }

  const offset = transform.offset ?? { x: 0, y: 0, z: 0 }
  const rotationDeg = transform.rotationDeg ?? { x: 0, y: 0, z: 0 }
  const offsetVector = toSceneVector(offset)
  target.position.copy(offsetVector)
  target.rotation.set(degToRad(rotationDeg.x), degToRad(rotationDeg.z), degToRad(rotationDeg.y))
  const scale = transform.scale ?? 1
  target.scale.setScalar(scale)
}

function fitObjectToGraph(object: THREE.Object3D, graph: NavigationGraphData): void {
  const objectBounds = new THREE.Box3().setFromObject(object)
  if (objectBounds.isEmpty()) {
    return
  }

  const graphCenter = toSceneVector({
    x: (graph.bounds.min.x + graph.bounds.max.x) / 2,
    y: (graph.bounds.min.y + graph.bounds.max.y) / 2,
    z: (graph.bounds.min.z + graph.bounds.max.z) / 2,
  })
  const objectCenter = objectBounds.getCenter(new THREE.Vector3())
  object.position.add(graphCenter.sub(objectCenter))
}

function colorFromHex(value: string): THREE.Color {
  return new THREE.Color(value)
}

function rememberOpacity(material: THREE.Material | THREE.Material[]): void {
  const materials = Array.isArray(material) ? material : [material]
  for (const item of materials) {
    item.transparent = true
    item.userData.baseOpacity = item.opacity
    item.userData.baseDepthWrite = item.depthWrite
    item.userData.baseDepthTest = item.depthTest
  }
}

function liftOverlayMaterial(material: THREE.Material | THREE.Material[]): void {
  const materials = Array.isArray(material) ? material : [material]
  for (const item of materials) {
    item.depthTest = false
    item.depthWrite = false
    item.userData.baseDepthTest = false
    item.userData.baseDepthWrite = false
  }
}

function setObjectOpacity(object: THREE.Object3D, opacity: number): void {
  object.traverse((child: THREE.Object3D) => {
    const mesh = child as THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>
    if (!mesh.material) {
      return
    }

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    for (const material of materials) {
      const baseOpacity = typeof material.userData.baseOpacity === 'number' ? material.userData.baseOpacity : 1
      const baseDepthTest =
        typeof material.userData.baseDepthTest === 'boolean' ? material.userData.baseDepthTest : material.depthTest
      const baseDepthWrite =
        typeof material.userData.baseDepthWrite === 'boolean' ? material.userData.baseDepthWrite : material.depthWrite
      material.opacity = baseOpacity * opacity
      material.transparent = opacity < 1 || baseOpacity < 1
      material.depthTest = baseDepthTest
      material.depthWrite = baseDepthWrite && opacity >= 0.95
    }
  })
}

function sampleLevelColor(level: string): THREE.Color {
  if (level.startsWith('F1')) {
    return new THREE.Color('#f4a261')
  }
  if (level.startsWith('F3')) {
    return new THREE.Color('#2ec4b6')
  }
  if (level.startsWith('F4')) {
    return new THREE.Color('#8ecae6')
  }
  if (level.includes('STAIR')) {
    return new THREE.Color('#ffd166')
  }
  return new THREE.Color('#b8c0ff')
}

function classifySpecialNodeType(nodeType: string): SpecialNodeKind | null {
  const normalized = nodeType.toLowerCase()
  if (normalized.includes('fare_gate')) {
    return 'fare_gate'
  }
  if (normalized.includes('entrance')) {
    return 'entrance'
  }
  if (normalized.includes('elevator')) {
    return 'elevator'
  }
  if (normalized.includes('escalator')) {
    return 'escalator'
  }
  if (normalized.includes('stair')) {
    return 'stair'
  }
  return null
}

function sampleSpecialNodeColor(kind: SpecialNodeKind): THREE.Color {
  switch (kind) {
    case 'fare_gate':
      return new THREE.Color('#ff595e')
    case 'entrance':
      return new THREE.Color('#7ae582')
    case 'stair':
      return new THREE.Color('#ff9f1c')
    case 'escalator':
      return new THREE.Color('#ffd166')
    case 'elevator':
      return new THREE.Color('#b8c0ff')
  }
}

function sampleAnchorColor(type: string): THREE.Color {
  const normalized = type.toUpperCase()
  if (normalized.includes('EXIT')) {
    return new THREE.Color('#ff595e')
  }
  if (normalized.includes('ENTRANCE')) {
    return new THREE.Color('#7ae582')
  }
  if (normalized.includes('PLATFORM')) {
    return new THREE.Color('#ffd166')
  }
  if (normalized.includes('FARE_GATE')) {
    return new THREE.Color('#ff595e')
  }
  return new THREE.Color('#8ecae6')
}

function anchorTypeSize(type: string): number {
  const normalized = type.toUpperCase()
  if (normalized.includes('PLATFORM')) {
    return 0.38
  }
  if (normalized.includes('ENTRANCE') || normalized.includes('EXIT')) {
    return 0.68
  }
  if (normalized.includes('FARE_GATE')) {
    return 0.78
  }
  return 0.55
}

function getAdaptiveGraphStep(edgeCount: number, configuredStep: number): number {
  if (edgeCount > 24000) {
    return Math.max(configuredStep, 4)
  }
  if (edgeCount > 14000) {
    return Math.max(configuredStep, 3)
  }
  if (edgeCount > 7000) {
    return Math.max(configuredStep, 2)
  }
  return Math.max(1, configuredStep)
}

function getAdaptiveNodeStep(nodeCount: number): number {
  if (nodeCount > 18000) {
    return 6
  }
  if (nodeCount > 9000) {
    return 4
  }
  if (nodeCount > 4000) {
    return 3
  }
  return 2
}

function specialNodePointSize(kind: SpecialNodeKind): number {
  switch (kind) {
    case 'fare_gate':
      return 1.75
    case 'entrance':
      return 1.45
    case 'stair':
      return 1.15
    case 'escalator':
      return 1.3
    case 'elevator':
      return 1.4
  }
}

function focusFromBounds(bounds: THREE.Box3, multiplier = 1): { position: THREE.Vector3; target: THREE.Vector3 } {
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const span = Math.max(size.x, size.y, size.z, 10) * multiplier
  return {
    position: center.clone().add(new THREE.Vector3(span * 0.75, span * 0.62, span * 0.95)),
    target: center,
  }
}

export class DemoScene {
  private readonly container: HTMLElement

  private readonly config: DemoConfig

  private readonly scene = new THREE.Scene()
  private readonly renderer: THREE.WebGLRenderer
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private readonly gltfLoader = new GLTFLoader()
  private readonly plyLoader = new PLYLoader()
  private readonly clock = new THREE.Clock()
  private readonly resizeObserver: ResizeObserver
  private readonly root = new THREE.Group()
  private readonly layers = {
    model: new THREE.Group(),
    context: new THREE.Group(),
    pointCloud: new THREE.Group(),
    movement: new THREE.Group(),
    graph: new THREE.Group(),
    route: new THREE.Group(),
    anchors: new THREE.Group(),
    evaluation: new THREE.Group(),
    simulation: new THREE.Group(),
  }

  private bundle: DemoDataBundle | null = null
  private cameraTween: CameraTween | null = null
  private animationHandle = 0
  private activeStage: StageDefinition['id'] = 1
  private autoRotateEnabled = false
  private modelStoreyGroups: THREE.Group[] = []
  private currentScenario: SimulationScenario | null = null
  private simulationPointSets: Partial<Record<'normal' | 'elderly', SimulationPointSet>> = {}
  private simulationMarkerGroup: THREE.Group | null = null
  private simulationReplanMaterial: THREE.SpriteMaterial | null = null
  private simulationReplanEventsByFrame = new Map<number, Set<string>>()
  private playing = false
  private playbackStartMs = 0
  private playbackStartTime = 0
  private status: SceneStatus = {
    routeLabel: 'No route loaded',
    simulationLabel: 'No simulation loaded',
    modelMode: 'placeholder',
    pointCloudMode: 'placeholder',
    movementMode: 'graph-fallback',
    simulationMode: 'illustrative',
  }

  public constructor(
    container: HTMLElement,
    config: DemoConfig,
  ) {
    this.container = container
    this.config = config
    this.scene.background = null
    this.scene.fog = new THREE.Fog(colorFromHex(this.config.visuals.backgroundBottom), 110, 420)

    this.camera = new THREE.PerspectiveCamera(
      this.config.camera.fov,
      1,
      this.config.camera.near,
      this.config.camera.far,
    )
    this.camera.position.copy(toSceneVector(this.config.camera.position))

    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' })
    this.renderer.setPixelRatio(1)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.setClearColor(0x000000, 0)
    this.container.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = this.config.controls.enableDamping
    this.controls.dampingFactor = this.config.controls.dampingFactor
    this.controls.minDistance = this.config.controls.minDistance
    this.controls.maxDistance = this.config.controls.maxDistance
    this.controls.target.copy(toSceneVector(this.config.camera.target))
    this.controls.autoRotateSpeed = this.config.visuals.autoRotateSpeed
    this.controls.autoRotate = false

    this.scene.add(this.root)
    for (const group of Object.values(this.layers)) {
      this.root.add(group)
    }

    this.addLights()
    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(this.container)
    this.resize()
    this.animate()
  }

  public async hydrate(
    bundle: DemoDataBundle,
    onWarning: (message: string) => void,
  ): Promise<void> {
    this.bundle = bundle
    this.clearAllLayers()

    this.layers.context.add(this.buildContextShell(bundle.graph))
    this.buildGraph(bundle.graph)
    this.buildAnchors(bundle)
    this.buildMovement(bundle.movement)
    await Promise.all([
      this.loadModel(bundle, onWarning),
      this.loadPointCloud(bundle, onWarning),
    ])

    const defaultRoute =
      bundle.routes.routes.find((route) => route.id === bundle.routes.defaultRouteId) ?? bundle.routes.routes[0] ?? null
    const defaultScenario =
      bundle.simulation.scenarios.find((scenario) => scenario.id === bundle.simulation.defaultScenarioId) ??
      bundle.simulation.scenarios[0] ??
      null

    if (defaultRoute) {
      this.setRoute(defaultRoute)
    }

    if (defaultScenario) {
      this.setScenario(defaultScenario)
    }

    this.setStage(1)
  }

  public setStage(stageId: StageDefinition['id']): void {
    this.activeStage = stageId
    this.applyModelStagePose()
    this.applyStageVisibility()
    this.startCameraTween(stageId)
  }

  public setRoute(route: DemoRoute): void {
    this.status.routeLabel = route.label
    this.layers.route.clear()

    const polyline = route.polyline ?? []
    if (polyline.length < 2) {
      return
    }

    const points = polyline.map((point) => toSceneVector(point))
    const curve = new THREE.CatmullRomCurve3(points)
    const geometry = new THREE.TubeGeometry(curve, Math.max(24, points.length * 4), 0.45, 12, false)
    const material = new THREE.MeshStandardMaterial({
      color: colorFromHex(this.config.visuals.route),
      emissive: colorFromHex(this.config.visuals.route),
      emissiveIntensity: 0.15,
      opacity: 0.96,
    })
    rememberOpacity(material)
    liftOverlayMaterial(material)
    const routeMesh = new THREE.Mesh(geometry, material)
    routeMesh.renderOrder = 18
    this.layers.route.add(routeMesh)

    const startMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 20, 20),
      new THREE.MeshStandardMaterial({ color: '#7ae582', emissive: '#7ae582', emissiveIntensity: 0.2 }),
    )
    const endMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 20, 20),
      new THREE.MeshStandardMaterial({ color: '#ff595e', emissive: '#ff595e', emissiveIntensity: 0.2 }),
    )
    liftOverlayMaterial(startMarker.material)
    liftOverlayMaterial(endMarker.material)
    startMarker.renderOrder = 19
    endMarker.renderOrder = 19
    startMarker.position.copy(points[0])
    endMarker.position.copy(points[points.length - 1])
    this.layers.route.add(startMarker, endMarker)

    if (this.activeStage === 5 || this.activeStage === 6) {
      this.startCameraTween(this.activeStage)
    }
  }

  public setScenario(scenario: SimulationScenario): void {
    this.currentScenario = scenario
    this.status.simulationLabel = scenario.label
    this.status.simulationMode = scenario.kind
    this.layers.simulation.clear()
    this.layers.evaluation.clear()
    this.simulationPointSets = {}
    this.simulationMarkerGroup = null
    this.simulationReplanEventsByFrame = this.indexReplanEvents(scenario)
    this.playing = false

    const maxAgents = scenario.frames.reduce(
      (result, frame) => {
        const counts = { normal: 0, elderly: 0 }
        for (const agent of frame.agents) {
          counts[normalizeSimulationAgentType(scenario.agentMeta?.[agent.id]?.agentType)] += 1
        }

        return {
          normal: Math.max(result.normal, counts.normal),
          elderly: Math.max(result.elderly, counts.elderly),
        }
      },
      { normal: 0, elderly: 0 },
    )
    if (maxAgents.normal === 0 && maxAgents.elderly === 0) {
      return
    }

    const normalSet = this.createSimulationPointSet(
      sampleSimulationAgentColor(scenario, 'normal'),
      this.config.visuals.simulationPointSize * SIMULATION_POINT_SCALE,
      maxAgents.normal,
      23,
    )
    const elderlySet = this.createSimulationPointSet(
      sampleSimulationAgentColor(scenario, 'elderly'),
      this.config.visuals.simulationPointSize * SIMULATION_ELDERLY_SCALE,
      maxAgents.elderly,
      25,
    )

    if (normalSet) {
      this.simulationPointSets.normal = normalSet
      this.layers.simulation.add(normalSet.glow, normalSet.core)
    }
    if (elderlySet) {
      this.simulationPointSets.elderly = elderlySet
      this.layers.simulation.add(elderlySet.glow, elderlySet.core)
    }

    this.simulationMarkerGroup = new THREE.Group()
    this.layers.simulation.add(this.simulationMarkerGroup)
    this.renderSimulationFrame(0)
    this.buildEvaluationLayer(scenario)
  }

  public toggleSimulationPlayback(): boolean {
    if (!this.currentScenario || this.currentScenario.frames.length === 0) {
      return false
    }

    this.playing = !this.playing
    if (this.playing) {
      const playbackStartTime = this.preferredPlaybackStartTime(this.currentScenario)
      this.playbackStartMs = performance.now()
      this.playbackStartTime = playbackStartTime
      this.renderSimulationFrame(playbackStartTime)
    }
    return this.playing
  }

  public setAutoRotate(enabled: boolean): void {
    this.autoRotateEnabled = enabled
    this.controls.autoRotate = enabled
  }

  public toggleAutoRotate(): boolean {
    this.setAutoRotate(!this.autoRotateEnabled)
    return this.autoRotateEnabled
  }

  public beginGestureManipulation(): void {
    this.cameraTween = null
    this.controls.autoRotate = false
  }

  public updateGestureManipulation(deltaX: number, deltaY: number): void {
    this.cameraTween = null
    const offset = this.camera.position.clone().sub(this.controls.target)
    const spherical = new THREE.Spherical().setFromVector3(offset)
    spherical.theta -= deltaX * 0.0085
    spherical.phi = THREE.MathUtils.clamp(spherical.phi + deltaY * 0.0065, 0.18, Math.PI - 0.18)
    offset.setFromSpherical(spherical)
    this.camera.position.copy(this.controls.target.clone().add(offset))
    this.controls.update()
  }

  public adjustGestureZoom(direction: 1 | -1): void {
    this.cameraTween = null
    const offset = this.camera.position.clone().sub(this.controls.target)
    const currentDistance = offset.length()
    const nextDistance = THREE.MathUtils.clamp(
      currentDistance * (direction > 0 ? 0.82 : 1.18),
      this.config.controls.minDistance,
      this.config.controls.maxDistance,
    )
    offset.setLength(nextDistance)
    this.camera.position.copy(this.controls.target.clone().add(offset))
    this.controls.update()
  }

  public endGestureManipulation(): void {
  }

  public resetCamera(): void {
    this.startCameraTween(this.activeStage)
  }

  public getStatus(): SceneStatus {
    return this.status
  }

  public destroy(): void {
    cancelAnimationFrame(this.animationHandle)
    this.resizeObserver.disconnect()
    this.controls.dispose()
    this.renderer.dispose()
    this.container.removeChild(this.renderer.domElement)
  }

  private clearAllLayers(): void {
    for (const [, group] of Object.entries(this.layers) as Array<[keyof typeof this.layers, THREE.Group]>) {
      group.clear()
    }
    this.modelStoreyGroups = []
  }

  private addLights(): void {
    const ambient = new THREE.AmbientLight('#f8f9fa', 1.7)
    const directional = new THREE.DirectionalLight('#ffffff', 1.1)
    directional.position.set(45, 80, 20)
    const accent = new THREE.DirectionalLight('#4cc9f0', 0.45)
    accent.position.set(-30, 25, -50)
    this.scene.add(ambient, directional, accent)
  }

  private async loadModel(bundle: DemoDataBundle, onWarning: (message: string) => void): Promise<void> {
    const wrapper = new THREE.Group()
    this.layers.model.add(wrapper)

    try {
      const gltf = await this.gltfLoader.loadAsync(resolvePublicAssetUrl(bundle.config.assets.bimModel.path))
      gltf.scene.traverse((child: THREE.Object3D) => {
        const mesh = child as THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>
        if (mesh.material) {
          rememberOpacity(mesh.material)
        }
      })
      applyTransform(gltf.scene, bundle.config.assets.bimModel.transform)
      gltf.scene.updateMatrixWorld(true)

      const explodedModel = this.buildExplodedAssetModel(gltf.scene, bundle.graph)
      wrapper.add(explodedModel ?? gltf.scene)
      this.status.modelMode = 'asset'
    } catch (error) {
      if (!bundle.config.placeholderFlags.model) {
        throw error
      }

      onWarning(
        `Could not load ${bundle.config.assets.bimModel.path}. Stage 1 uses exploded placeholder floor plates derived from graph footprints.`,
      )
      wrapper.add(this.buildPlaceholderModel(bundle.graph))
      this.status.modelMode = 'placeholder'
    }
  }

  private async loadPointCloud(bundle: DemoDataBundle, onWarning: (message: string) => void): Promise<void> {
    const wrapper = new THREE.Group()
    this.layers.pointCloud.add(wrapper)

    try {
      const pointCloudUrl = resolvePublicAssetUrl(bundle.config.assets.pointCloud.path)
      const response = await fetch(pointCloudUrl, { cache: 'no-cache' })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const buffer = await response.arrayBuffer()
      const signature = new TextDecoder().decode(buffer.slice(0, 16)).trimStart().toLowerCase()
      if (!signature.startsWith('ply')) {
        throw new Error('Asset did not return a valid PLY header.')
      }

      const geometry = this.plyLoader.parse(buffer)
      const positions = geometry.getAttribute('position')
      if (!positions || positions.count === 0) {
        throw new Error('PLY geometry contained no vertices.')
      }

      geometry.computeBoundingSphere()
      const material = new THREE.PointsMaterial({
        color: colorFromHex(bundle.config.visuals.pointCloud),
        size: bundle.config.visuals.pointSize,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.92,
      })
      rememberOpacity(material)
      const points = new THREE.Points(geometry, material)
      applyTransform(points, bundle.config.assets.pointCloud.transform)
      fitObjectToGraph(points, bundle.graph)
      wrapper.add(points)
      this.status.pointCloudMode = 'asset'
    } catch (error) {
      if (!bundle.config.placeholderFlags.pointCloud) {
        throw error
      }

      onWarning(
        `Could not load ${bundle.config.assets.pointCloud.path}. Stage 3 uses aggregated BIM-derived density tiles sampled from navigation-supporting nodes.`,
      )
      wrapper.add(this.buildPlaceholderPointCloud(bundle.graph))
      this.status.pointCloudMode = 'placeholder'
    }
  }

  private buildPlaceholderModel(graph: NavigationGraphData): THREE.Object3D {
    const group = new THREE.Group()
    const storeys = this.buildStoreyDescriptors(graph)

    for (const storey of storeys) {
      const storeyGroup = this.createStoreyGuide(storey, {
        fillOpacity: 0.72,
        outlineOpacity: 0.18,
      })
      const baseY = storey.z
      this.registerModelStoreyGroup(storeyGroup, baseY, baseY + storey.order * 4.6)
      group.add(storeyGroup)
    }

    return group
  }

  private buildStoreyDescriptors(graph: NavigationGraphData): StoreyDescriptor[] {
    const grouped = new Map<string, typeof graph.nodes>()

    for (const node of graph.nodes) {
      if (!isWalkableStoreyNode(node)) {
        continue
      }

      const bucket = grouped.get(node.level) ?? []
      bucket.push(node)
      grouped.set(node.level, bucket)
    }

    return Array.from(grouped.entries())
      .map(([level, nodes]) => ({
        level,
        z: nodes.reduce((sum, node) => sum + node.z, 0) / Math.max(1, nodes.length),
        footprint: buildConvexHull(nodes.map((node) => ({ x: node.x, y: node.y }))),
      }))
      .filter((storey) => storey.footprint.length >= 3)
      .sort((left, right) => left.z - right.z)
      .map((storey, order) => ({ ...storey, order }))
  }

  private createStoreyGuide(
    storey: StoreyDescriptor,
    options: { fillOpacity: number; outlineOpacity: number },
  ): THREE.Group {
    const thickness = 0.65
    const group = new THREE.Group()
    const rectangle = buildBoundingRectangle(storey.footprint)
    const shape = new THREE.Shape(rectangle.map((point) => new THREE.Vector2(point.x, -point.y)))
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false,
    })
    geometry.rotateX(-Math.PI / 2)

    const material = new THREE.MeshStandardMaterial({
      color: sampleLevelColor(storey.level),
      metalness: 0.04,
      roughness: 0.84,
      transparent: true,
      opacity: options.fillOpacity,
    })
    rememberOpacity(material)
    group.add(new THREE.Mesh(geometry, material))

    const outlineGeometry = new THREE.BufferGeometry().setFromPoints(
      rectangle.map((point) => new THREE.Vector3(point.x, thickness + 0.08, -point.y)),
    )
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: '#f8f9fa',
      transparent: true,
      opacity: options.outlineOpacity,
    })
    rememberOpacity(outlineMaterial)
    group.add(new THREE.LineLoop(outlineGeometry, outlineMaterial))

    return group
  }

  private buildExplodedAssetModel(sourceScene: THREE.Object3D, graph: NavigationGraphData): THREE.Group | null {
    const storeys = this.buildStoreyDescriptors(graph)
    if (storeys.length === 0) {
      return null
    }

    const root = new THREE.Group()
    const buckets = new Map<string, THREE.Group>()
    for (const storey of storeys) {
      const bucket = new THREE.Group()
      const baseY = 0
      this.registerModelStoreyGroup(bucket, baseY, storey.order * 4.6)
      buckets.set(storey.level, bucket)
      root.add(bucket)
    }

    const meshes: Array<THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>> = []
    let meshCount = 0
    sourceScene.updateMatrixWorld(true)
    sourceScene.traverse((child: THREE.Object3D) => {
      const mesh = child as THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>
      if (!mesh.isMesh) {
        return
      }

      meshes.push(mesh)
    })

    for (const mesh of meshes) {
      const bounds = new THREE.Box3().setFromObject(mesh)
      if (bounds.isEmpty()) {
        continue
      }

      const targetStorey = storeys.reduce((best, candidate) =>
        Math.abs(candidate.z - bounds.getCenter(new THREE.Vector3()).y) < Math.abs(best.z - bounds.getCenter(new THREE.Vector3()).y)
          ? candidate
          : best,
      )

      buckets.get(targetStorey.level)?.attach(mesh)
      mesh.matrixAutoUpdate = false
      mesh.updateMatrix()
      meshCount += 1
    }

    return meshCount > 0 ? root : null
  }

  private registerModelStoreyGroup(group: THREE.Group, baseY: number, explodedY: number): void {
    group.position.y = baseY
    group.userData.baseY = baseY
    group.userData.explodedY = explodedY
    this.modelStoreyGroups.push(group)
  }

  private applyModelStagePose(): void {
    const explodeRatio = this.activeStage === 1 ? 1 : this.activeStage === 2 ? 0.5 : 0
    for (const group of this.modelStoreyGroups) {
      const baseY = typeof group.userData.baseY === 'number' ? group.userData.baseY : group.position.y
      const explodedY = typeof group.userData.explodedY === 'number' ? group.userData.explodedY : group.position.y
      group.position.y = THREE.MathUtils.lerp(baseY, explodedY, explodeRatio)
    }
  }

  private buildContextShell(graph: NavigationGraphData): THREE.Object3D {
    const shell = new THREE.Group()
    const storeys = this.buildStoreyDescriptors(graph)

    for (const storey of storeys) {
      const outline = this.createStoreyOutline(storey)
      outline.position.y = storey.z + 0.02
      shell.add(outline)
    }

    return shell
  }

  private createStoreyOutline(storey: StoreyDescriptor): THREE.Group {
    const group = new THREE.Group()
    const rectangle = buildBoundingRectangle(storey.footprint)
    const topHeight = 0.42

    const outlineGeometry = new THREE.BufferGeometry().setFromPoints(
      rectangle.map((point) => new THREE.Vector3(point.x, topHeight, -point.y)),
    )
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: sampleLevelColor(storey.level),
      transparent: true,
      opacity: 0.34,
    })
    rememberOpacity(outlineMaterial)
    liftOverlayMaterial(outlineMaterial)

    const outline = new THREE.LineLoop(outlineGeometry, outlineMaterial)
    outline.renderOrder = 2
    group.add(outline)

    const postSegments: number[] = []
    for (const point of rectangle) {
      postSegments.push(point.x, 0, -point.y, point.x, topHeight, -point.y)
    }

    const postGeometry = new THREE.BufferGeometry()
    postGeometry.setAttribute('position', new THREE.Float32BufferAttribute(postSegments, 3))
    const postMaterial = new THREE.LineBasicMaterial({
      color: sampleLevelColor(storey.level),
      transparent: true,
      opacity: 0.18,
    })
    rememberOpacity(postMaterial)
    liftOverlayMaterial(postMaterial)

    const posts = new THREE.LineSegments(postGeometry, postMaterial)
    posts.renderOrder = 2
    group.add(posts)

    return group
  }

  private buildPlaceholderPointCloud(graph: NavigationGraphData): THREE.Object3D {
    const samples: TileFieldSample[] = graph.nodes
      .filter((node) => isWalkableStoreyNode(node) && node.usable !== false)
      .map((node) => ({
        x: node.x,
        y: node.y,
        z: node.z,
        level: node.level,
      }))

    return (
      this.createTileField(samples, {
        cellSize: 1.8,
        minHeight: 0.18,
        heightStep: 0.1,
        maxHeight: 0.74,
        opacity: 0.72,
        renderOrder: 5,
        palette: 'pointCloud',
      }) ?? new THREE.Group()
    )
  }

  private buildMovement(movement: MovementGeometryData): void {
    this.layers.movement.clear()
    this.status.movementMode = movement.surfaces.some((surface) => surface.source === 'asset')
      ? 'asset'
      : 'graph-fallback'
    const step = Math.max(1, this.config.visuals.movementRenderStep)
    const samples: TileFieldSample[] = []

    for (const surface of movement.surfaces) {
      const sampledPoints = surface.points.filter((_, index) => index % step === 0 || index === surface.points.length - 1)
      sampledPoints.forEach((point) => {
        samples.push({
          x: point.x,
          y: point.y,
          z: point.z,
          level: surface.level,
        })
      })
    }

    const tileField = this.createTileField(samples, {
      cellSize: 1.45,
      minHeight: 0.12,
      heightStep: 0.05,
      maxHeight: 0.38,
      opacity: 0.88,
      renderOrder: 6,
      palette: 'movement',
    })

    if (tileField) {
      this.layers.movement.add(tileField)
    }
  }

  private createSolidPointCloud(
    positions: number[],
    color: THREE.Color,
    size: number,
    opacity: number,
    renderOrder: number,
  ): THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> | null {
    if (positions.length === 0) {
      return null
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))

    const material = new THREE.PointsMaterial({
      color,
      size,
      sizeAttenuation: true,
      transparent: true,
      opacity,
    })
    rememberOpacity(material)
    liftOverlayMaterial(material)

    const points = new THREE.Points(geometry, material)
    points.renderOrder = renderOrder
    return points
  }

  private createTileField(
    samples: TileFieldSample[],
    options: TileFieldOptions,
  ): THREE.InstancedMesh<THREE.BoxGeometry, THREE.MeshBasicMaterial> | null {
    if (samples.length === 0) {
      return null
    }

    const cells = new Map<
      string,
      { sumX: number; sumY: number; sumZ: number; count: number; level: string }
    >()

    for (const sample of samples) {
      const gridX = Math.round(sample.x / options.cellSize)
      const gridY = Math.round(sample.y / options.cellSize)
      const key = `${sample.level}:${gridX}:${gridY}`
      const cell = cells.get(key) ?? { sumX: 0, sumY: 0, sumZ: 0, count: 0, level: sample.level }
      cell.sumX += sample.x
      cell.sumY += sample.y
      cell.sumZ += sample.z
      cell.count += 1
      cells.set(key, cell)
    }

    if (cells.size === 0) {
      return null
    }

    const geometry = new THREE.BoxGeometry(1, 1, 1)
    const material = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      transparent: true,
      opacity: options.opacity,
    })
    rememberOpacity(material)
    liftOverlayMaterial(material)

    const mesh = new THREE.InstancedMesh(geometry, material, cells.size)
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage)

    const matrix = new THREE.Matrix4()
    const quaternion = new THREE.Quaternion()
    const position = new THREE.Vector3()
    const scale = new THREE.Vector3()

    let index = 0
    for (const cell of cells.values()) {
      const avgX = cell.sumX / cell.count
      const avgY = cell.sumY / cell.count
      const avgZ = cell.sumZ / cell.count
      const height = Math.min(options.maxHeight, options.minHeight + Math.log2(cell.count + 1) * options.heightStep)
      const levelColor = sampleLevelColor(cell.level)
      const tileColor =
        options.palette === 'pointCloud'
          ? levelColor.clone().lerp(colorFromHex(this.config.visuals.pointCloud), 0.58)
          : levelColor.clone().lerp(colorFromHex(this.config.visuals.accentSoft), 0.24)

      position.copy(toSceneVector({ x: avgX, y: avgY, z: avgZ }))
      position.y += height * 0.5
      scale.set(options.cellSize * 0.84, height, options.cellSize * 0.84)
      matrix.compose(position, quaternion, scale)

      mesh.setMatrixAt(index, matrix)
      mesh.setColorAt(index, tileColor)
      index += 1
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true
    }
    mesh.renderOrder = options.renderOrder
    return mesh
  }

  private buildEvaluationLayer(scenario: SimulationScenario): void {
    this.layers.evaluation.clear()

    const comparison = this.resolveEvaluationScenarios(scenario)
    if (!comparison) {
      return
    }

    const deltaField = this.createEvaluationDeltaField(comparison.evaluated, comparison.baseline)
    if (deltaField) {
      this.layers.evaluation.add(deltaField)
    }
  }

  private resolveEvaluationScenarios(
    scenario: SimulationScenario,
  ): { baseline: SimulationScenario; evaluated: SimulationScenario } | null {
    if (!this.bundle) {
      return null
    }

    const baseline = this.bundle.simulation.scenarios.find((candidate) => isStaticScenario(candidate))
    if (!baseline) {
      return null
    }

    const evaluated = !isStaticScenario(scenario)
      ? scenario
      : this.bundle.simulation.scenarios.find((candidate) => !isStaticScenario(candidate) && candidate.frames.length > 0)

    if (!evaluated || evaluated.id === baseline.id) {
      return null
    }

    return { baseline, evaluated }
  }

  private collectScenarioDensity(scenario: SimulationScenario, cellSize: number): Map<string, ScenarioDensityCell> {
    const cells = new Map<string, ScenarioDensityCell>()
    const frameStep =
      scenario.frames.length > 1400 ? 6 : scenario.frames.length > 700 ? 4 : scenario.frames.length > 280 ? 2 : 1
    let sampledFrameCount = 0

    for (let frameIndex = 0; frameIndex < scenario.frames.length; frameIndex += frameStep) {
      const frame = scenario.frames[frameIndex]
      sampledFrameCount += 1

      for (const agent of frame.agents) {
        const gridX = Math.round(agent.x / cellSize)
        const gridY = Math.round(agent.y / cellSize)
        const gridZ = Math.round(agent.z / 3)
        const key = `${gridZ}:${gridX}:${gridY}`
        const cell = cells.get(key) ?? { sumX: 0, sumY: 0, sumZ: 0, samples: 0, density: 0 }
        cell.sumX += agent.x
        cell.sumY += agent.y
        cell.sumZ += agent.z
        cell.samples += 1
        cell.density += 1
        cells.set(key, cell)
      }
    }

    const normalization = Math.max(sampledFrameCount, 1)
    for (const cell of cells.values()) {
      cell.density /= normalization
    }

    return cells
  }

  private createEvaluationDeltaField(
    evaluated: SimulationScenario,
    baseline: SimulationScenario,
  ): THREE.InstancedMesh<THREE.BoxGeometry, THREE.MeshBasicMaterial> | null {
    const cellSize = 3.2
    const evaluatedCells = this.collectScenarioDensity(evaluated, cellSize)
    const baselineCells = this.collectScenarioDensity(baseline, cellSize)
    const keys = new Set([...evaluatedCells.keys(), ...baselineCells.keys()])
    const entries: Array<{ x: number; y: number; z: number; delta: number }> = []
    let maxAbsDelta = 0

    for (const key of keys) {
      const evaluatedCell = evaluatedCells.get(key)
      const baselineCell = baselineCells.get(key)
      const reference = evaluatedCell ?? baselineCell
      if (!reference) {
        continue
      }

      const delta = (evaluatedCell?.density ?? 0) - (baselineCell?.density ?? 0)
      maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta))
      entries.push({
        x: reference.sumX / reference.samples,
        y: reference.sumY / reference.samples,
        z: reference.sumZ / reference.samples,
        delta,
      })
    }

    if (entries.length === 0 || maxAbsDelta <= 0.01) {
      return null
    }

    const visibleEntries = entries.filter((entry) => Math.abs(entry.delta) >= Math.max(0.08, maxAbsDelta * 0.16))
    if (visibleEntries.length === 0) {
      return null
    }

    const geometry = new THREE.BoxGeometry(1, 1, 1)
    const material = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      transparent: true,
      opacity: 0.92,
    })
    rememberOpacity(material)
    liftOverlayMaterial(material)

    const mesh = new THREE.InstancedMesh(geometry, material, visibleEntries.length)
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage)

    const reliefLow = new THREE.Color('#8ecae6')
    const reliefHigh = new THREE.Color('#2ec4b6')
    const pressureLow = new THREE.Color('#ffd166')
    const pressureHigh = new THREE.Color('#ff595e')
    const matrix = new THREE.Matrix4()
    const quaternion = new THREE.Quaternion()
    const position = new THREE.Vector3()
    const scale = new THREE.Vector3()

    visibleEntries.forEach((entry, index) => {
      const normalized = THREE.MathUtils.clamp(Math.abs(entry.delta) / maxAbsDelta, 0, 1)
      const height = THREE.MathUtils.lerp(0.16, 1.35, normalized)
      const color =
        entry.delta < 0
          ? reliefLow.clone().lerp(reliefHigh, normalized)
          : pressureLow.clone().lerp(pressureHigh, normalized)

      position.copy(toSceneVector({ x: entry.x, y: entry.y, z: entry.z }))
      position.y += height * 0.5 + 0.08
      scale.set(cellSize * 0.78, height, cellSize * 0.78)
      matrix.compose(position, quaternion, scale)

      mesh.setMatrixAt(index, matrix)
      mesh.setColorAt(index, color)
    })

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true
    }
    mesh.renderOrder = 16
    return mesh
  }

  private buildGraph(graph: NavigationGraphData): void {
    this.layers.graph.clear()
    const step = getAdaptiveGraphStep(graph.edges.length, this.config.visuals.graphRenderStep)
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
    const basePositions: number[] = []
    const baseColors: number[] = []
    const accentPositions: number[] = []
    const accentColors: number[] = []

    graph.edges.forEach((edge, index) => {
      const source = nodes.get(edge.source)
      const target = nodes.get(edge.target)
      if (!source || !target) {
        return
      }

      const sourcePoint = toSceneVector(source)
      const targetPoint = toSceneVector(target)
      const specialKind = classifySpecialNodeType(edge.edgeType)
      const useAccentLayer = specialKind !== null

      if (!useAccentLayer && index % step !== 0) {
        return
      }

      const positions = useAccentLayer ? accentPositions : basePositions
      const colors = useAccentLayer ? accentColors : baseColors
      const color = useAccentLayer ? sampleSpecialNodeColor(specialKind) : sampleLevelColor(edge.level)

      positions.push(sourcePoint.x, sourcePoint.y, sourcePoint.z, targetPoint.x, targetPoint.y, targetPoint.z)
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b)
    })

    if (basePositions.length > 0) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(basePositions, 3))
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(baseColors, 3))

      const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.26,
      })
      rememberOpacity(material)
      liftOverlayMaterial(material)

      const edgeSegments = new THREE.LineSegments(geometry, material)
      edgeSegments.renderOrder = 8
      this.layers.graph.add(edgeSegments)
    }

    if (accentPositions.length > 0) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(accentPositions, 3))
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(accentColors, 3))

      const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
      })
      rememberOpacity(material)
      liftOverlayMaterial(material)

      const edgeSegments = new THREE.LineSegments(geometry, material)
      edgeSegments.renderOrder = 11
      this.layers.graph.add(edgeSegments)
    }

    const walkableNodeStep = getAdaptiveNodeStep(graph.nodes.length)
    const walkablePositions: number[] = []
    const walkableColors: number[] = []
    const specialNodes = new Map<SpecialNodeKind, NavigationNode[]>()

    graph.nodes.forEach((node, index) => {
      const specialKind = classifySpecialNodeType(node.nodeType)
      if (specialKind) {
        const bucket = specialNodes.get(specialKind) ?? []
        bucket.push(node)
        specialNodes.set(specialKind, bucket)
        return
      }

      if (node.usable === false || index % walkableNodeStep !== 0) {
        return
      }

      const point = toSceneVector(node)
      const color = sampleLevelColor(node.level)
      walkablePositions.push(point.x, point.y, point.z)
      walkableColors.push(color.r, color.g, color.b)
    })

    if (walkablePositions.length > 0) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(walkablePositions, 3))
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(walkableColors, 3))

      const material = new THREE.PointsMaterial({
        size: 0.28,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: 0.34,
      })
      rememberOpacity(material)
      liftOverlayMaterial(material)

      const points = new THREE.Points(geometry, material)
      points.renderOrder = 10
      this.layers.graph.add(points)
    }

    for (const [kind, bucket] of specialNodes) {
      const sampleStep = bucket.length > 140 ? 5 : bucket.length > 60 ? 3 : 1
      const positions: number[] = []
      for (let index = 0; index < bucket.length; index += sampleStep) {
        const point = toSceneVector(bucket[index])
        positions.push(point.x, point.y + 0.1, point.z)
      }

      if (positions.length === 0) {
        continue
      }

      const color = sampleSpecialNodeColor(kind)
      const glowGeometry = new THREE.BufferGeometry()
      glowGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      const glowMaterial = new THREE.PointsMaterial({
        color,
        size: specialNodePointSize(kind) + 0.85,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.2,
      })
      rememberOpacity(glowMaterial)
      liftOverlayMaterial(glowMaterial)
      const glowPoints = new THREE.Points(glowGeometry, glowMaterial)
      glowPoints.renderOrder = 12

      const coreGeometry = new THREE.BufferGeometry()
      coreGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      const coreMaterial = new THREE.PointsMaterial({
        color,
        size: specialNodePointSize(kind),
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.95,
      })
      rememberOpacity(coreMaterial)
      liftOverlayMaterial(coreMaterial)
      const corePoints = new THREE.Points(coreGeometry, coreMaterial)
      corePoints.renderOrder = 13

      this.layers.graph.add(glowPoints, corePoints)
    }
  }

  private buildAnchors(bundle: DemoDataBundle): void {
    this.layers.anchors.clear()
    const grouped = new Map<string, number[]>()

    for (const anchor of bundle.anchors.anchors) {
      const point = toSceneVector(anchor)
      const positions = grouped.get(anchor.type) ?? []
      positions.push(point.x, point.y + 0.18, point.z)
      grouped.set(anchor.type, positions)
    }

    for (const [type, positions] of grouped.entries()) {
      const color = sampleAnchorColor(type)
      const glow = this.createSolidPointCloud(positions, color, anchorTypeSize(type) * 1.8, 0.14, 14)
      const core = this.createSolidPointCloud(positions, color, anchorTypeSize(type), 0.86, 15)

      if (glow) {
        this.layers.anchors.add(glow)
      }
      if (core) {
        this.layers.anchors.add(core)
      }
    }
  }

  private applyStageVisibility(): void {
    const stageVisibility: Record<StageDefinition['id'], Record<keyof typeof this.layers, number>> = {
      1: { model: 1, context: 0, pointCloud: 0, movement: 0, graph: 0, route: 0, anchors: 0, evaluation: 0, simulation: 0 },
      2: { model: 0.24, context: 0.84, pointCloud: 0, movement: 0.14, graph: 0, route: 0, anchors: 0.42, evaluation: 0, simulation: 0 },
      3: { model: 0, context: 0.62, pointCloud: 0, movement: 1, graph: 0, route: 0, anchors: 0.2, evaluation: 0, simulation: 0 },
      4: { model: 0, context: 0.16, pointCloud: 0.16, movement: 0.12, graph: 1, route: 0, anchors: 0.32, evaluation: 0, simulation: 0 },
      5: { model: 0, context: 0.08, pointCloud: 0, movement: 0.08, graph: 0.48, route: 1, anchors: 0.18, evaluation: 0, simulation: 0.76 },
      6: { model: 0, context: 0.02, pointCloud: 0, movement: 0.06, graph: 0.56, route: 0.42, anchors: 0.12, evaluation: 1, simulation: 0.26 },
    }

    for (const [key, group] of Object.entries(this.layers) as Array<[keyof typeof this.layers, THREE.Group]>) {
      const opacity = stageVisibility[this.activeStage][key]
      group.visible = opacity > 0.01
      setObjectOpacity(group, opacity)
    }
  }

  private startCameraTween(stageId: StageDefinition['id']): void {
    if (!this.bundle) {
      return
    }

    const focusBox =
      (stageId === 5 || stageId === 6) && this.layers.route.children.length > 0
        ? new THREE.Box3().setFromObject(this.layers.route)
        : new THREE.Box3().setFromObject(this.root)

    const { position, target } = focusFromBounds(
      focusBox,
      stageId === 2 ? 0.92 : stageId === 4 ? 1.08 : stageId === 6 ? 1.28 : 1,
    )

    const offset = new THREE.Vector3()
    if (stageId === 3) {
      offset.set(-16, 10, -18)
    } else if (stageId === 4) {
      offset.set(5, 28, 6)
    } else if (stageId === 5) {
      offset.set(9, 18, 14)
    } else if (stageId === 6) {
      offset.set(18, 12, 12)
    }

    this.cameraTween = {
      startPosition: this.camera.position.clone(),
      endPosition: position.add(offset),
      startTarget: this.controls.target.clone(),
      endTarget: target,
      startedAt: performance.now(),
      duration: this.config.visuals.transitionMs,
    }
  }

  private renderSimulationFrame(elapsedSeconds: number): void {
    if (!this.currentScenario) {
      return
    }

    const frames = this.currentScenario.frames
    if (frames.length === 0) {
      return
    }

    const clampedTime = Math.min(
      this.currentScenario.timeline.end,
      Math.max(this.currentScenario.timeline.start, elapsedSeconds),
    )
    const step = Math.max(this.currentScenario.timeline.step, 0.0001)
    const frameIndex = Math.min(
      frames.length - 1,
      Math.floor((clampedTime - this.currentScenario.timeline.start) / step),
    )
    const frame = frames[frameIndex]
    const recentReplanAgents = this.collectRecentReplanAgents(frameIndex)

    const drawCounts = { normal: 0, elderly: 0 }
    frame.agents.forEach((agent) => {
      const type = normalizeSimulationAgentType(this.currentScenario?.agentMeta?.[agent.id]?.agentType)
      const set = this.simulationPointSets[type]
      if (!set) {
        return
      }

      const pointIndex = drawCounts[type]
      drawCounts[type] += 1
      const point = toSceneVector(agent)
      set.positions[pointIndex * 3] = point.x
      set.positions[pointIndex * 3 + 1] = point.y
      set.positions[pointIndex * 3 + 2] = point.z
    })

    ;(['normal', 'elderly'] as const).forEach((type) => {
      const set = this.simulationPointSets[type]
      if (!set) {
        return
      }

      set.geometry.setDrawRange(0, drawCounts[type])
      const positionAttribute = set.geometry.getAttribute('position') as THREE.BufferAttribute
      positionAttribute.needsUpdate = true
    })

    this.updateSimulationMarkers(frame, clampedTime, recentReplanAgents)
  }

  private createSimulationPointSet(
    color: THREE.Color,
    size: number,
    maxCount: number,
    renderOrder: number,
  ): SimulationPointSet | null {
    if (maxCount <= 0) {
      return null
    }

    const positions = new Float32Array(maxCount * 3)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setDrawRange(0, maxCount)

    const glowMaterial = new THREE.PointsMaterial({
      color,
      size: size * SIMULATION_GLOW_SCALE,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.24,
      blending: THREE.AdditiveBlending,
    })
    rememberOpacity(glowMaterial)
    liftOverlayMaterial(glowMaterial)
    const glow = new THREE.Points(geometry, glowMaterial)
    glow.renderOrder = renderOrder

    const coreMaterial = new THREE.PointsMaterial({
      color,
      size,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.98,
    })
    rememberOpacity(coreMaterial)
    liftOverlayMaterial(coreMaterial)
    const core = new THREE.Points(geometry, coreMaterial)
    core.renderOrder = renderOrder + 1

    return { geometry, positions, glow, core }
  }

  private preferredPlaybackStartTime(scenario: SimulationScenario): number {
    const firstReplanTime = scenario.replanEvents?.[0]?.t
    if (typeof firstReplanTime !== 'number' || !Number.isFinite(firstReplanTime)) {
      return scenario.timeline.start
    }

    return Math.max(scenario.timeline.start, firstReplanTime - REPLAN_PREVIEW_LEAD_SECONDS)
  }

  private indexReplanEvents(scenario: SimulationScenario): Map<number, Set<string>> {
    const indexed = new Map<number, Set<string>>()
    const step = Math.max(scenario.timeline.step, 0.0001)

    for (const event of scenario.replanEvents ?? []) {
      const frameIndex = Math.max(0, Math.round((event.t - scenario.timeline.start) / step))
      const frameEvents = indexed.get(frameIndex) ?? new Set<string>()
      frameEvents.add(event.agentId)
      indexed.set(frameIndex, frameEvents)
    }

    return indexed
  }

  private collectRecentReplanAgents(frameIndex: number): Set<string> {
    const recent = new Set<string>()
    for (let offset = 0; offset <= REPLAN_MARKER_LIFETIME_FRAMES; offset += 1) {
      const eventAgents = this.simulationReplanEventsByFrame.get(frameIndex - offset)
      if (!eventAgents) {
        continue
      }

      for (const agentId of eventAgents) {
        recent.add(agentId)
      }
    }

    return recent
  }

  private ensureSimulationReplanMaterial(): THREE.SpriteMaterial {
    if (this.simulationReplanMaterial) {
      return this.simulationReplanMaterial
    }

    const material = new THREE.SpriteMaterial({
      map: createReplanMarkerTexture(),
      transparent: true,
      opacity: 0.97,
    })
    rememberOpacity(material)
    liftOverlayMaterial(material)
    this.simulationReplanMaterial = material
    return material
  }

  private updateSimulationMarkers(
    frame: { agents: Array<{ id: string; x: number; y: number; z: number }> },
    currentTime: number,
    recentReplanAgents: Set<string>,
  ): void {
    if (!this.simulationMarkerGroup) {
      return
    }

    this.simulationMarkerGroup.clear()
    if (recentReplanAgents.size === 0) {
      return
    }

    const material = this.ensureSimulationReplanMaterial()
    const pulse = 1 + Math.sin(currentTime * 9) * 0.14
    const bob = Math.sin(currentTime * 7) * 0.22

    for (const agent of frame.agents) {
      if (!recentReplanAgents.has(agent.id)) {
        continue
      }

      const sprite = new THREE.Sprite(material)
      const point = toSceneVector(agent)
      sprite.position.set(point.x, point.y + REPLAN_MARKER_HEIGHT + bob, point.z)
      sprite.scale.set(5.2 * pulse, 5.2 * pulse, 1)
      sprite.renderOrder = 28
      this.simulationMarkerGroup.add(sprite)
    }
  }

  private updateCameraTween(now: number): void {
    if (!this.cameraTween) {
      return
    }

    const progress = Math.min(1, (now - this.cameraTween.startedAt) / this.cameraTween.duration)
    const eased = 1 - Math.pow(1 - progress, 3)
    this.camera.position.lerpVectors(this.cameraTween.startPosition, this.cameraTween.endPosition, eased)
    this.controls.target.lerpVectors(this.cameraTween.startTarget, this.cameraTween.endTarget, eased)

    if (progress >= 1) {
      this.cameraTween = null
    }
  }

  private animate = (): void => {
    this.animationHandle = requestAnimationFrame(this.animate)
    const now = performance.now()
    this.updateCameraTween(now)

    if (this.playing && this.currentScenario) {
      const elapsed = this.playbackStartTime + (now - this.playbackStartMs) / 1000
      if (elapsed > this.currentScenario.timeline.end) {
        this.playbackStartMs = now
        this.playbackStartTime = this.currentScenario.timeline.start
        this.renderSimulationFrame(this.currentScenario.timeline.start)
      } else {
        this.renderSimulationFrame(elapsed)
      }
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
    this.clock.getDelta()
  }

  private resize(): void {
    const width = this.container.clientWidth || 1
    const height = this.container.clientHeight || 1
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }
}
