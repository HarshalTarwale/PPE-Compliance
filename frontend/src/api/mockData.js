/**
 * ISOLATED DEVELOPMENT MOCK DATA
 *
 * NOTE: Production code consumes the live FastAPI backend via REST and WebSockets.
 * This mock data is strictly isolated for offline visual verification and development testing.
 */

export const ISOLATED_DEV_MOCK_STATUS = {
  is_active: true,
  source: 'Webcam #0 (1080p)',
  total_tracked: 3,
  compliant_count: 2,
  non_compliant_count: 1,
  persons: [
    {
      track_id: 'track_1',
      employee_id: 'EMP101',
      employee_name: 'Dr. Jane Chen',
      hardhat: true,
      safety_vest: true,
      compliance_status: 'COMPLIANT',
      missing_items: [],
      detection_confidence: 0.942,
      bbox: [120, 80, 280, 420],
      is_compliant: true,
      face_similarity: 0.781,
    },
    {
      track_id: 'track_2',
      employee_id: 'EMP102',
      employee_name: 'Marcus Vance',
      hardhat: true,
      safety_vest: true,
      compliance_status: 'COMPLIANT',
      missing_items: [],
      detection_confidence: 0.891,
      bbox: [320, 95, 470, 430],
      is_compliant: true,
      face_similarity: 0.812,
    },
    {
      track_id: 'track_3',
      employee_id: 'UNKNOWN',
      employee_name: 'UNKNOWN',
      hardhat: false,
      safety_vest: true,
      compliance_status: 'NON_COMPLIANT',
      missing_items: ['Hardhat'],
      detection_confidence: 0.875,
      bbox: [500, 110, 620, 440],
      is_compliant: false,
      face_similarity: null,
    },
  ],
  telemetry: {
    fps: 31.4,
    total_ms: 31.8,
    detection_ms: 11.5,
    association_ms: 2.3,
    recognition_ms: 16.2,
    smoothing_ms: 1.8,
  },
  timestamp: new Date().toISOString(),
};

export const ISOLATED_DEV_MOCK_EMPLOYEES = [
  {
    employee_id: 'EMP101',
    employee_name: 'Dr. Jane Chen',
    registered_at: '2026-09-20T10:15:00Z',
    sample_count: 4,
    embedding_dim: 128,
    metadata: { role: 'Lab Safety Officer', site: 'Workshop Area A' },
  },
  {
    employee_id: 'EMP102',
    employee_name: 'Marcus Vance',
    registered_at: '2026-09-22T14:30:00Z',
    sample_count: 3,
    embedding_dim: 128,
    metadata: { role: 'Field Engineer', site: 'Fabrication Hall' },
  },
  {
    employee_id: 'EMP103',
    employee_name: 'Sarah Connor',
    registered_at: '2026-09-24T09:00:00Z',
    sample_count: 5,
    embedding_dim: 128,
    metadata: { role: 'Technician', site: 'Assembly Station 2' },
  },
];
