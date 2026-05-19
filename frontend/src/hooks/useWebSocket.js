/**
 * useWebSocket — Real-time WebSocket connection with auto-reconnect.
 * Connects to /ws/live, parses JSON, calls onMessage(data) every 5s.
 *
 * Resilient to React Strict Mode double-mounting and re-renders.
 * Uses a ref for the callback to avoid re-creating the connection
 * when the parent component re-renders.
 */

import { useEffect, useRef } from 'react'

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`
const RECONNECT_MS = 3000

export default function useWebSocket(onMessage) {
  // Store callback in a ref so WS handlers always see the latest
  // without causing the effect to re-run
  const callbackRef = useRef(onMessage)
  callbackRef.current = onMessage

  useEffect(() => {
    let mounted = true
    let ws = null
    let reconnectTimer = null

    function connect() {
      if (!mounted) return

      try {
        ws = new WebSocket(WS_URL)

        ws.onopen = () => {
          console.log('[WS] connected')
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            callbackRef.current(data)
          } catch (e) {
            console.error('[WS] parse error', e)
          }
        }

        ws.onerror = (err) => {
          console.warn('[WS] error', err)
          ws.close()
        }

        ws.onclose = () => {
          console.log('[WS] closed, reconnecting in', RECONNECT_MS, 'ms')
          ws = null
          if (mounted) {
            reconnectTimer = setTimeout(connect, RECONNECT_MS)
          }
        }
      } catch (e) {
        console.error('[WS] connection failed', e)
        if (mounted) {
          reconnectTimer = setTimeout(connect, RECONNECT_MS)
        }
      }
    }

    connect()

    return () => {
      mounted = false
      clearTimeout(reconnectTimer)
      if (ws) {
        ws.onclose = null  // prevent reconnect on intentional close
        ws.close()
      }
    }
  }, [])  // Empty deps — connect once, never re-create
}
