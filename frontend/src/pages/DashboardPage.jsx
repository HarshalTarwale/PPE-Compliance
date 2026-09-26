import React from 'react';
import { Users, ShieldCheck, ShieldAlert, Cpu, Activity, Clock } from 'lucide-react';
import StatCard from '../components/StatCard';
import LiveMonitoring from '../components/LiveMonitoring';
import EmployeeTable from '../components/EmployeeTable';
import ViolationAlert from '../components/ViolationAlert';

export default function DashboardPage({
  monitoringStatus,
  onStartStream,
  onStopStream,
  isStreamLoading,
}) {
  const {
    is_active = false,
    source = null,
    total_tracked = 0,
    compliant_count = 0,
    non_compliant_count = 0,
    persons = [],
    telemetry = {},
  } = monitoringStatus || {};

  const violations = persons.filter((p) => !p.is_compliant);
  const complianceRate =
    total_tracked > 0 ? ((compliant_count / total_tracked) * 100).toFixed(0) : 100;

  return (
    <div>
      {/* Safety Violation Alert Bar */}
      <ViolationAlert violations={violations} />

      {/* KPI Metric Readouts */}
      <div className="metrics-grid">
        <StatCard
          label="People Detected"
          value={total_tracked}
          sub={is_active ? 'Active tracking in FOV' : 'Monitoring standby'}
          accent="telemetry"
          icon={Users}
        />
        <StatCard
          label="Compliant Personnel"
          value={compliant_count}
          sub={`${complianceRate}% compliance rate`}
          accent="compliant"
          icon={ShieldCheck}
        />
        <StatCard
          label="Non-Compliant Violations"
          value={non_compliant_count}
          sub={non_compliant_count > 0 ? 'Immediate action required' : 'Zero violations'}
          accent={non_compliant_count > 0 ? 'violation' : 'neutral'}
          icon={ShieldAlert}
        />
        <StatCard
          label="Inference Telemetry"
          value={`${telemetry.fps ? telemetry.fps.toFixed(1) : '0.0'} FPS`}
          sub={`Total: ${telemetry.total_ms ? telemetry.total_ms.toFixed(1) : '0.0'} ms | Det: ${telemetry.detection_ms ? telemetry.detection_ms.toFixed(1) : '0.0'} ms`}
          accent="telemetry"
          icon={Activity}
        />
      </div>

      {/* Main Monitoring & Status Grid */}
      <div className="dashboard-split">
        <LiveMonitoring
          isActive={is_active}
          source={source}
          persons={persons}
          telemetry={telemetry}
          totalTracked={total_tracked}
          compliantCount={compliant_count}
          nonCompliantCount={non_compliant_count}
          onStart={onStartStream}
          onStop={onStopStream}
          isLoading={isStreamLoading}
        />

        <EmployeeTable persons={persons} />
      </div>
    </div>
  );
}
