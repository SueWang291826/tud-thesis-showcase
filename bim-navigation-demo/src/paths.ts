export function resolvePublicAssetUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path
  }

  const normalizedPath = path.replace(/^\/+/, '')
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.href)
  return new URL(normalizedPath, baseUrl).toString()
}
