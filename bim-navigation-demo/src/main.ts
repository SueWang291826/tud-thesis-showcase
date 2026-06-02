import './style.css'
import { bootstrapDemo } from './app'

const root = document.querySelector<HTMLDivElement>('#app')

if (!root) {
  throw new Error('Unable to find #app root element.')
}

void bootstrapDemo(root)
