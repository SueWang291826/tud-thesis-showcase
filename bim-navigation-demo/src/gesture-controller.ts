import { resolvePublicAssetUrl } from './paths'
import type { DemoConfig, GestureCommand, GestureManipulationEvent } from './types'

type CommandCallback = (command: GestureCommand) => void

type ManipulationCallback = (event: GestureManipulationEvent) => void

type StatusCallback = (message: string) => void

type GestureMode = 'mediapipe' | 'motion-fallback' | 'pointer-fallback'

interface MotionCentroid {
  x: number
  y: number
}

interface LandmarkPoint extends MotionCentroid {
  z: number
}

interface GestureRecognizerLike {
  recognizeForVideo(video: HTMLVideoElement, timestampMs: number): {
    gestures?: Array<Array<{ categoryName?: string; score?: number }>>
    landmarks?: LandmarkPoint[][]
  }
  close?: () => void
}

interface VisionModule {
  FilesetResolver: {
    forVisionTasks(wasmPath: string): Promise<unknown>
  }
  GestureRecognizer: {
    createFromOptions(
      fileset: unknown,
      options: {
        baseOptions: { modelAssetPath: string }
        runningMode: 'VIDEO'
        numHands: number
      },
    ): Promise<GestureRecognizerLike>
  }
}

export class GestureController {
  private readonly gestureMap = new Map<string, GestureCommand>([
    ['Open_Palm', 'next'],
    ['Pointing_Up', 'prev'],
    ['Victory', 'toggleSimulation'],
    ['Thumb_Up', 'toggleAutoRotate'],
    ['ILoveYou', 'zoomIn'],
    ['Thumb_Down', 'zoomOut'],
  ])

  private recognizer: GestureRecognizerLike | null = null
  private stream: MediaStream | null = null
  private video: HTMLVideoElement | null = null
  private frameHandle = 0
  private lastTriggerAt = 0
  private lastZoomAt = 0
  private running = false
  private mode: GestureMode | null = null
  private analysisCanvas: HTMLCanvasElement | null = null
  private analysisContext: CanvasRenderingContext2D | null = null
  private previousFrame: Uint8Array | null = null
  private lastMotionCentroid: MotionCentroid | null = null
  private lastGrabPoint: MotionCentroid | null = null
  private grabActive = false
  private pointerStart: MotionCentroid | null = null
  private pointerDownAt = 0
  private pointerFallbackCleanup: (() => void) | null = null

  private readonly config: DemoConfig

  private readonly onCommand: CommandCallback

  private readonly onManipulation: ManipulationCallback

  private readonly onStatus: StatusCallback

  private readonly previewHost?: HTMLElement

  public constructor(
    config: DemoConfig,
    onCommand: CommandCallback,
    onManipulation: ManipulationCallback,
    onStatus: StatusCallback,
    previewHost?: HTMLElement,
  ) {
    this.config = config
    this.onCommand = onCommand
    this.onManipulation = onManipulation
    this.onStatus = onStatus
    this.previewHost = previewHost
  }

  public get isRunning(): boolean {
    return this.running
  }

  public async start(): Promise<void> {
    if (this.running) {
      return
    }

    if (!this.config.assets.mediapipe.enabled) {
      throw new Error('Gesture control is disabled in public/config/demo-config.json.')
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      this.initializePointerFallback('This browser does not expose webcam access through getUserMedia.')
      this.running = true
      return
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 360,
        },
        audio: false,
      })

      this.video = document.createElement('video')
      this.video.autoplay = true
      this.video.muted = true
      this.video.playsInline = true
      this.video.srcObject = this.stream
      this.attachPreviewVideo(this.video)
      await this.video.play()
    } catch (error) {
      this.initializePointerFallback(error instanceof Error ? error.message : String(error))
      this.running = true
      return
    }

    try {
      const vision = (await import('@mediapipe/tasks-vision')) as VisionModule
      const fileset = await vision.FilesetResolver.forVisionTasks(
        resolvePublicAssetUrl(this.config.assets.mediapipe.wasmPath),
      )

      this.recognizer = await vision.GestureRecognizer.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: resolvePublicAssetUrl(this.config.assets.mediapipe.modelAssetPath),
        },
        runningMode: 'VIDEO',
        numHands: 1,
      })
      this.mode = 'mediapipe'
      this.onStatus(
        'Gesture control active. Open palm: next. Pointing up: previous. Closed fist plus hand motion: orbit the view. I love you: zoom in. Thumb down: zoom out. Victory: play or pause. Thumb up: auto-rotate.',
      )
    } catch (error) {
      this.mode = 'motion-fallback'
      this.initializeMotionFallback()
      this.onStatus(
        `MediaPipe assets unavailable. Motion fallback active. Quick swipes trigger commands and slower motion orbits the view. ${error instanceof Error ? error.message : String(error)}`,
      )
    }

    this.running = true
    this.frameHandle = requestAnimationFrame(this.tick)
  }

  public stop(): void {
    if (this.frameHandle) {
      cancelAnimationFrame(this.frameHandle)
      this.frameHandle = 0
    }

    this.running = false
    if (this.mode) {
      this.releaseManipulation(this.mode)
    }
    this.mode = null
    this.previousFrame = null
    this.lastMotionCentroid = null
    this.analysisContext = null
    this.analysisCanvas = null
    this.lastGrabPoint = null
    this.grabActive = false
    this.pointerStart = null
    this.pointerDownAt = 0

    if (this.pointerFallbackCleanup) {
      this.pointerFallbackCleanup()
      this.pointerFallbackCleanup = null
    }

    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop()
      }
      this.stream = null
    }

    if (this.video) {
      this.video.pause()
      this.detachPreviewVideo(this.video)
      this.video = null
    } else if (this.previewHost) {
      this.previewHost.replaceChildren()
    }

    if (this.recognizer?.close) {
      this.recognizer.close()
    }
    this.recognizer = null

    this.onStatus('Gesture control stopped. Keyboard and mouse controls remain active.')
  }

  private readonly tick = (): void => {
    if (!this.running) {
      return
    }

    if (this.mode === 'pointer-fallback') {
      return
    }

    if (!this.video) {
      return
    }

    if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const now = performance.now()
      if (this.mode === 'mediapipe' && this.recognizer) {
        const result = this.recognizer.recognizeForVideo(this.video, now)
        const candidate = result.gestures?.[0]?.[0]
        const categoryName =
          candidate?.categoryName &&
          typeof candidate.score === 'number' &&
          candidate.score >= this.config.assets.mediapipe.minScore
            ? candidate.categoryName
            : null

        if (categoryName === 'Closed_Fist' && result.landmarks?.[0]?.length) {
          this.updateManipulationFromPoint(
            'mediapipe',
            this.computeLandmarkCentroid(result.landmarks[0]),
            'Closed-fist grab active. Move your hand to orbit the station view.',
            540,
            420,
          )
        } else {
          this.releaseManipulation('mediapipe')

          if (categoryName && now - this.lastTriggerAt >= this.config.assets.mediapipe.cooldownMs) {
            const command = this.gestureMap.get(categoryName)
            if (command) {
              if (command === 'zoomIn' || command === 'zoomOut') {
                this.triggerZoomCommand(
                  command,
                  command === 'zoomIn' ? 'Gesture zoom: in.' : 'Gesture zoom: out.',
                  now,
                )
              } else {
                this.triggerCommand(command, `Gesture recognized: ${categoryName}.`, now)
              }
            }
          }
        }
      } else if (this.mode === 'motion-fallback') {
        this.processMotionFrame(now)
      }
    }

    this.frameHandle = requestAnimationFrame(this.tick)
  }

  private attachPreviewVideo(video: HTMLVideoElement): void {
    if (this.previewHost) {
      this.previewHost.replaceChildren(video)
      return
    }

    video.style.position = 'fixed'
    video.style.right = '12px'
    video.style.bottom = '12px'
    video.style.width = '180px'
    video.style.borderRadius = '12px'
    video.style.border = '1px solid rgba(255,255,255,0.2)'
    video.style.transform = 'scaleX(-1)'
    video.style.zIndex = '12'
    document.body.appendChild(video)
  }

  private detachPreviewVideo(video: HTMLVideoElement): void {
    if (this.previewHost) {
      this.previewHost.replaceChildren()
      return
    }

    video.remove()
  }

  private initializeMotionFallback(): void {
    this.analysisCanvas = document.createElement('canvas')
    this.analysisCanvas.width = 128
    this.analysisCanvas.height = 96
    this.analysisContext = this.analysisCanvas.getContext('2d', { willReadFrequently: true })
    if (!this.analysisContext) {
      throw new Error('Unable to create an analysis canvas for fallback gesture detection.')
    }
  }

  private processMotionFrame(now: number): void {
    if (!this.video || !this.analysisCanvas || !this.analysisContext) {
      return
    }

    const width = this.analysisCanvas.width
    const height = this.analysisCanvas.height
    this.analysisContext.save()
    this.analysisContext.clearRect(0, 0, width, height)
    this.analysisContext.translate(width, 0)
    this.analysisContext.scale(-1, 1)
    this.analysisContext.drawImage(this.video, 0, 0, width, height)
    this.analysisContext.restore()
    const data = this.analysisContext.getImageData(0, 0, width, height).data
    const currentFrame = new Uint8Array(width * height)
    const previousFrame = this.previousFrame
    let activePixels = 0
    let sumX = 0
    let sumY = 0

    for (let pixelIndex = 0; pixelIndex < width * height; pixelIndex += 1) {
      const sourceIndex = pixelIndex * 4
      const grayscale = Math.round(
        data[sourceIndex] * 0.299 + data[sourceIndex + 1] * 0.587 + data[sourceIndex + 2] * 0.114,
      )
      currentFrame[pixelIndex] = grayscale

      if (previousFrame && Math.abs(grayscale - previousFrame[pixelIndex]) > 30) {
        const x = pixelIndex % width
        const y = Math.floor(pixelIndex / width)
        activePixels += 1
        sumX += x
        sumY += y
      }
    }

    this.previousFrame = currentFrame
    if (!previousFrame || activePixels < 170) {
      this.releaseManipulation('motion-fallback')
      this.lastMotionCentroid = null
      return
    }

    const centroid = {
      x: sumX / activePixels,
      y: sumY / activePixels,
    }

    if (this.lastMotionCentroid) {
      const deltaX = centroid.x - this.lastMotionCentroid.x
      const deltaY = centroid.y - this.lastMotionCentroid.y
      const absX = Math.abs(deltaX)
      const absY = Math.abs(deltaY)
      let command: GestureCommand | null = null
      let label = ''

      if (absX > 18 && absX > absY * 1.25) {
        command = deltaX > 0 ? 'next' : 'prev'
        label = deltaX > 0 ? 'Motion gesture: swipe right.' : 'Motion gesture: swipe left.'
      } else if (absY > 16 && absY > absX * 1.2) {
        command = deltaY < 0 ? 'toggleAutoRotate' : 'toggleSimulation'
        label = deltaY < 0 ? 'Motion gesture: swipe up.' : 'Motion gesture: swipe down.'
      }

      if (command && now - this.lastTriggerAt >= this.config.assets.mediapipe.cooldownMs) {
        this.releaseManipulation('motion-fallback')
        this.triggerCommand(command, label, now)
      } else if (
        !command &&
        now - this.lastTriggerAt >= this.config.assets.mediapipe.cooldownMs &&
        (absX > 2.5 || absY > 2.5)
      ) {
        this.updateManipulationFromPoint(
          'motion-fallback',
          centroid,
          'Motion fallback grab active. Move your hand to orbit the station view.',
          3.6,
          3.6,
        )
      }
    }

    this.lastMotionCentroid = centroid
  }

  private computeLandmarkCentroid(points: LandmarkPoint[]): MotionCentroid {
    const stablePoints = [0, 5, 9, 13, 17]
      .map((index) => points[index] ?? points[0])
      .filter((point): point is LandmarkPoint => Boolean(point))

    const total = stablePoints.reduce(
      (sum, point) => ({
        x: sum.x + point.x,
        y: sum.y + point.y,
      }),
      { x: 0, y: 0 },
    )

    return {
      x: total.x / stablePoints.length,
      y: total.y / stablePoints.length,
    }
  }

  private updateManipulationFromPoint(
    source: GestureManipulationEvent['source'],
    point: MotionCentroid,
    statusMessage: string,
    scaleX = 1,
    scaleY = 1,
  ): void {
    if (!this.grabActive) {
      this.grabActive = true
      this.lastGrabPoint = point
      this.onManipulation({
        phase: 'start',
        source,
        deltaX: 0,
        deltaY: 0,
      })
      this.onStatus(statusMessage)
      return
    }

    if (!this.lastGrabPoint) {
      this.lastGrabPoint = point
      return
    }

    const deltaX = (point.x - this.lastGrabPoint.x) * scaleX
    const deltaY = (point.y - this.lastGrabPoint.y) * scaleY
    this.lastGrabPoint = point

    if (Math.abs(deltaX) < 0.001 && Math.abs(deltaY) < 0.001) {
      return
    }

    this.onManipulation({
      phase: 'move',
      source,
      deltaX,
      deltaY,
    })
  }

  private releaseManipulation(source: GestureManipulationEvent['source'], statusMessage?: string): void {
    if (!this.grabActive) {
      this.lastGrabPoint = null
      return
    }

    this.grabActive = false
    this.lastGrabPoint = null
    this.onManipulation({
      phase: 'end',
      source,
      deltaX: 0,
      deltaY: 0,
    })

    if (statusMessage) {
      this.onStatus(statusMessage)
    }
  }

  private triggerCommand(command: GestureCommand, label: string, timestampMs: number): void {
    if (this.grabActive && this.mode) {
      this.releaseManipulation(this.mode)
    }

    this.lastTriggerAt = timestampMs
    this.onCommand(command)
    this.onStatus(label)
  }

  private triggerZoomCommand(command: 'zoomIn' | 'zoomOut', label: string, timestampMs: number): void {
    if (this.grabActive && this.mode) {
      this.releaseManipulation(this.mode)
    }

    if (timestampMs - this.lastZoomAt < 180) {
      return
    }

    this.lastZoomAt = timestampMs
    this.onCommand(command)
    this.onStatus(label)
  }

  private initializePointerFallback(reason: string): void {
    if (!this.previewHost) {
      throw new Error(reason)
    }

    this.mode = 'pointer-fallback'
    const panel = document.createElement('div')
    panel.className = 'gesture-pointer-fallback'
    panel.innerHTML = [
      '<strong>Gesture simulator</strong>',
      '<span>Quick drag right: next</span>',
      '<span>Quick drag left: previous</span>',
      '<span>Quick drag up or down: auto-rotate or simulation</span>',
      '<span>Hold briefly, then drag: orbit the station view</span>',
      '<span>Wheel or trackpad pinch here: zoom</span>',
    ].join('')
    this.previewHost.replaceChildren(panel)

    const onPointerDown = (event: PointerEvent): void => {
      this.pointerStart = { x: event.clientX, y: event.clientY }
      this.pointerDownAt = performance.now()
      try {
        this.previewHost?.setPointerCapture(event.pointerId)
      } catch {
        // Synthetic pointer events used in tests do not always establish a capturable pointer.
      }
    }

    const onPointerMove = (event: PointerEvent): void => {
      if (!this.pointerStart) {
        return
      }

      const heldLongEnough = performance.now() - this.pointerDownAt >= 180
      const travelled = Math.hypot(event.clientX - this.pointerStart.x, event.clientY - this.pointerStart.y)
      if (!heldLongEnough || travelled < 10) {
        return
      }

      this.updateManipulationFromPoint(
        'pointer-fallback',
        { x: event.clientX, y: event.clientY },
        'Swipe simulator grab active. Drag to orbit the station view.',
      )
      event.preventDefault()
    }

    const onPointerUp = (event: PointerEvent): void => {
      if (!this.pointerStart) {
        return
      }

      if (this.grabActive) {
        this.releaseManipulation('pointer-fallback', 'Swipe simulator: scene grab released.')
        this.pointerStart = null
        this.pointerDownAt = 0
        return
      }

      const now = performance.now()
      if (now - this.lastTriggerAt < this.config.assets.mediapipe.cooldownMs) {
        this.pointerStart = null
        this.pointerDownAt = 0
        return
      }

      const deltaX = event.clientX - this.pointerStart.x
      const deltaY = event.clientY - this.pointerStart.y
      const absX = Math.abs(deltaX)
      const absY = Math.abs(deltaY)
      let command: GestureCommand | null = null
      let label = ''

      if (absX > 36 && absX > absY * 1.2) {
        command = deltaX > 0 ? 'next' : 'prev'
        label = deltaX > 0 ? 'Swipe simulator: right.' : 'Swipe simulator: left.'
      } else if (absY > 36 && absY > absX * 1.2) {
        command = deltaY < 0 ? 'toggleAutoRotate' : 'toggleSimulation'
        label = deltaY < 0 ? 'Swipe simulator: up.' : 'Swipe simulator: down.'
      }

      if (command) {
        this.triggerCommand(command, label, now)
      }

      this.pointerStart = null
      this.pointerDownAt = 0
    }

    const onWheel = (event: WheelEvent): void => {
      const now = performance.now()
      const command = event.deltaY < 0 ? 'zoomIn' : 'zoomOut'
      this.triggerZoomCommand(
        command,
        command === 'zoomIn' ? 'Pinch simulator: zoom in.' : 'Pinch simulator: zoom out.',
        now,
      )
      event.preventDefault()
    }

    const onPointerCancel = (): void => {
      this.releaseManipulation('pointer-fallback')
      this.pointerStart = null
      this.pointerDownAt = 0
    }

    this.previewHost.addEventListener('pointerdown', onPointerDown)
    this.previewHost.addEventListener('pointermove', onPointerMove)
    this.previewHost.addEventListener('pointerup', onPointerUp)
    this.previewHost.addEventListener('wheel', onWheel)
    this.previewHost.addEventListener('pointercancel', onPointerCancel)
    this.pointerFallbackCleanup = () => {
      this.previewHost?.removeEventListener('pointerdown', onPointerDown)
      this.previewHost?.removeEventListener('pointermove', onPointerMove)
      this.previewHost?.removeEventListener('pointerup', onPointerUp)
      this.previewHost?.removeEventListener('wheel', onWheel)
      this.previewHost?.removeEventListener('pointercancel', onPointerCancel)
      this.previewHost?.replaceChildren()
    }

    this.onStatus(`Camera unavailable. Swipe, hold-drag, and pinch simulator active in the gesture panel. ${reason}`)
  }
}
