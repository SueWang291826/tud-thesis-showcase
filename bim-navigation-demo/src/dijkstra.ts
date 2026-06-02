import type {
  ComputedRoute,
  DemoRoute,
  NavigationEdge,
  NavigationGraphData,
  NavigationNode,
} from './types'

interface QueueItem {
  cost: number
  nodeId: string
}

interface Neighbor {
  edge: NavigationEdge
  nodeId: string
}

function buildNodeLookup(nodes: NavigationNode[]): Map<string, NavigationNode> {
  return new Map(nodes.map((node) => [node.id, node]))
}

function buildAdjacency(edges: NavigationEdge[]): Map<string, Neighbor[]> {
  const adjacency = new Map<string, Neighbor[]>()

  for (const edge of edges) {
    const forward = adjacency.get(edge.source) ?? []
    forward.push({ edge, nodeId: edge.target })
    adjacency.set(edge.source, forward)

    const reverse = adjacency.get(edge.target) ?? []
    reverse.push({ edge, nodeId: edge.source })
    adjacency.set(edge.target, reverse)
  }

  return adjacency
}

function push(queue: QueueItem[], item: QueueItem): void {
  queue.push(item)
  let index = queue.length - 1

  while (index > 0) {
    const parent = Math.floor((index - 1) / 2)
    if (queue[parent].cost <= queue[index].cost) {
      break
    }

    ;[queue[parent], queue[index]] = [queue[index], queue[parent]]
    index = parent
  }
}

function pop(queue: QueueItem[]): QueueItem | undefined {
  if (queue.length === 0) {
    return undefined
  }

  const top = queue[0]
  const tail = queue.pop()

  if (queue.length > 0 && tail) {
    queue[0] = tail
    let index = 0

    while (true) {
      const left = index * 2 + 1
      const right = index * 2 + 2
      let smallest = index

      if (left < queue.length && queue[left].cost < queue[smallest].cost) {
        smallest = left
      }

      if (right < queue.length && queue[right].cost < queue[smallest].cost) {
        smallest = right
      }

      if (smallest === index) {
        break
      }

      ;[queue[index], queue[smallest]] = [queue[smallest], queue[index]]
      index = smallest
    }
  }

  return top
}

export function computeShortestPath(
  graph: NavigationGraphData,
  startNodeId: string,
  endNodeId: string,
): ComputedRoute | null {
  const nodeLookup = buildNodeLookup(graph.nodes)
  const adjacency = buildAdjacency(graph.edges)

  if (!nodeLookup.has(startNodeId) || !nodeLookup.has(endNodeId)) {
    return null
  }

  const distances = new Map<string, number>([[startNodeId, 0]])
  const previous = new Map<string, string>()
  const previousEdge = new Map<string, NavigationEdge>()
  const queue: QueueItem[] = [{ cost: 0, nodeId: startNodeId }]
  const visited = new Set<string>()

  while (queue.length > 0) {
    const current = pop(queue)
    if (!current || visited.has(current.nodeId)) {
      continue
    }

    if (current.nodeId === endNodeId) {
      break
    }

    visited.add(current.nodeId)

    for (const neighbor of adjacency.get(current.nodeId) ?? []) {
      const tentativeCost = current.cost + neighbor.edge.travelTime
      const knownCost = distances.get(neighbor.nodeId) ?? Number.POSITIVE_INFINITY

      if (tentativeCost < knownCost) {
        distances.set(neighbor.nodeId, tentativeCost)
        previous.set(neighbor.nodeId, current.nodeId)
        previousEdge.set(neighbor.nodeId, neighbor.edge)
        push(queue, { cost: tentativeCost, nodeId: neighbor.nodeId })
      }
    }
  }

  if (!distances.has(endNodeId)) {
    return null
  }

  const nodeIds: string[] = []
  const edgeTypes: Record<string, number> = {}
  const levelSet = new Set<string>()
  let totalLength2d = 0
  let totalLength3d = 0
  let totalTravelTime = 0
  let cursor = endNodeId

  while (true) {
    nodeIds.push(cursor)
    if (cursor === startNodeId) {
      break
    }

    const edge = previousEdge.get(cursor)
    if (!edge) {
      return null
    }

    totalLength2d += edge.length2d
    totalLength3d += edge.length3d
    totalTravelTime += edge.travelTime
    edgeTypes[edge.edgeType] = (edgeTypes[edge.edgeType] ?? 0) + 1
    levelSet.add(edge.level)

    const parent = previous.get(cursor)
    if (!parent) {
      return null
    }

    cursor = parent
  }

  nodeIds.reverse()

  const polyline = nodeIds
    .map((nodeId) => nodeLookup.get(nodeId))
    .filter((node): node is NavigationNode => Boolean(node))
    .map((node) => ({ x: node.x, y: node.y, z: node.z }))

  return {
    nodeIds,
    polyline,
    totalTravelTime,
    totalLength2d,
    totalLength3d,
    edgeTypes,
    levelsVisited: Array.from(levelSet.values()),
  }
}

export function ensureRouteGeometry(
  route: DemoRoute,
  graph: NavigationGraphData,
): DemoRoute {
  if (route.snappedNodeIds && route.polyline && route.polyline.length >= 2) {
    return route
  }

  const computed = computeShortestPath(graph, route.originNodeId, route.destinationNodeId)
  if (!computed) {
    return route
  }

  return {
    ...route,
    sourceKind: 'derived-dijkstra',
    snappedNodeIds: computed.nodeIds,
    polyline: computed.polyline,
    metrics: {
      ...route.metrics,
      totalTravelTime:
        typeof route.metrics.totalTravelTime === 'number'
          ? route.metrics.totalTravelTime
          : computed.totalTravelTime,
      totalLength2d:
        typeof route.metrics.totalLength2d === 'number'
          ? route.metrics.totalLength2d
          : computed.totalLength2d,
      totalLength3d:
        typeof route.metrics.totalLength3d === 'number'
          ? route.metrics.totalLength3d
          : computed.totalLength3d,
      edgeTypes:
        typeof route.metrics.edgeTypes === 'object' && route.metrics.edgeTypes !== null
          ? route.metrics.edgeTypes
          : computed.edgeTypes,
      levelsVisited:
        Array.isArray(route.metrics.levelsVisited) && route.metrics.levelsVisited.length > 0
          ? route.metrics.levelsVisited
          : computed.levelsVisited,
    },
  }
}
