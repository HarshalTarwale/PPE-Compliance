/**
 * REST API client for PPE Compliance FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // Ignore json parse error
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse(res);
}

export async function fetchEmployees() {
  const res = await fetch(`${API_BASE}/employees`);
  return handleResponse(res);
}

export async function registerEmployee(formData) {
  const res = await fetch(`${API_BASE}/employees`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse(res);
}

export async function deleteEmployee(employeeId) {
  const res = await fetch(`${API_BASE}/employees/${encodeURIComponent(employeeId)}`, {
    method: 'DELETE',
  });
  return handleResponse(res);
}

export async function fetchMonitoringStatus() {
  const res = await fetch(`${API_BASE}/monitoring/status`);
  return handleResponse(res);
}

export async function startMonitoring(source = 0) {
  const res = await fetch(`${API_BASE}/monitoring/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  });
  return handleResponse(res);
}

export async function stopMonitoring() {
  const res = await fetch(`${API_BASE}/monitoring/stop`, {
    method: 'POST',
  });
  return handleResponse(res);
}
