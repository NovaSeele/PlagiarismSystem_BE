from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
import time
from schemas import LogEntry

router = APIRouter()

# Global storage for active WebSocket connections
active_connections: List[WebSocket] = []

# Global log buffer to store the most recent logs
log_buffer = []
MAX_BUFFER_SIZE = 1000  # Maximum number of log entries to store


@router.get("/api/logs")
async def get_logs():
    """
    REST API endpoint to get all logs from the buffer
    """
    return log_buffer


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for streaming logs to clients
    """
    await websocket.accept()
    active_connections.append(websocket)

    # Send the current log buffer to the new client
    for log in log_buffer:
        await websocket.send_text(json.dumps(log))
        # Add a very small delay to prevent WebSocket overload
        await asyncio.sleep(0.001)

    try:
        # Keep the connection open
        while True:
            # Wait for potential client messages (like clear logs)
            data = await websocket.receive_text()

            # Handle client commands if needed
            if data == "clear_logs":
                log_buffer.clear()
                # Send confirmation to client
                await websocket.send_text(
                    json.dumps({"type": "system", "message": "Logs cleared"})
                )

    except WebSocketDisconnect:
        # Remove connection when client disconnects
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_log(log_entry: Dict[str, Any]):
    """
    Broadcast a log entry to all connected clients
    """
    # Add to buffer (with size limit)
    if len(log_buffer) >= MAX_BUFFER_SIZE:
        log_buffer.pop(0)  # Remove oldest log
    log_buffer.append(log_entry)

    # Send to all active connections
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(log_entry))
        except Exception:
            # Connection might be closed but not properly removed
            if connection in active_connections:
                active_connections.remove(connection)


# Function that will be called from other modules to send logs
def send_log(module: str, message: str, level: str = "info"):
    """
    Send log entry to all connected WebSocket clients

    Args:
        module (str): The module name (e.g., "BERT", "LSA/LDA", "FastText")
        message (str): The log message
        level (str): Log level (info, warning, error)
    """
    log_entry = LogEntry(
        module=module,
        message=message,
        level=level,
        timestamp=time.time(),  # Use standard time.time() instead of asyncio time
    ).dict()

    # Use asyncio to run the broadcast function
    try:
        # Try to get the existing event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context with a running loop
                asyncio.create_task(broadcast_log(log_entry))
            else:
                # If loop exists but not running
                loop.run_until_complete(broadcast_log(log_entry))
        except RuntimeError:
            # Create a new event loop if none is available
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(broadcast_log(log_entry))
            # No need to close the loop as it might be needed again
    except Exception as e:
        # Last resort - at least save to buffer
        if len(log_buffer) >= MAX_BUFFER_SIZE:
            log_buffer.pop(0)
        log_buffer.append(log_entry)
        print(f"Error in send_log: {e}")
