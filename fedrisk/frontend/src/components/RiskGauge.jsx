import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, AlertCircle } from 'lucide-react';

export default function RiskGauge({ score = 0, percentage = 0, category = 'Low Risk', recommendation = '' }) {
  const normalizedPct = Math.min(Math.max(percentage, 0), 100);

  // SVG Gauge calculations
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  // Semi-circle arc (180 deg)
  const strokeDashoffset = circumference - (normalizedPct / 100) * (circumference / 2);

  // Dynamic Color
  let gaugeColor = '#10B981';
  let Icon = CheckCircle;
  let bgClass = 'badge-success';

  if (category === 'Critical Risk' || normalizedPct >= 75) {
    gaugeColor = '#EF4444';
    Icon = ShieldAlert;
    bgClass = 'badge-danger';
  } else if (category === 'High Risk' || normalizedPct >= 50) {
    gaugeColor = '#F97316';
    Icon = AlertTriangle;
    bgClass = 'badge-warning';
  } else if (category === 'Moderate Risk' || normalizedPct >= 25) {
    gaugeColor = '#F59E0B';
    Icon = AlertCircle;
    bgClass = 'badge-warning';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', justifyContent: 'space-between' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative', paddingTop: '10px' }}>
        <svg width="220" height="130" viewBox="0 0 200 120">
          {/* Background Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="16"
            strokeLinecap="round"
          />
          {/* Progress Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={gaugeColor}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={circumference / 2}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s ease' }}
          />
        </svg>

        {/* Center Percentage Display */}
        <div style={{ position: 'absolute', bottom: '15px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <span style={{ fontSize: '2.4rem', fontWeight: '800', fontFamily: 'Outfit', color: '#F8FAFC', lineHeight: 1 }}>
            {normalizedPct.toFixed(1)}%
          </span>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '2px' }}>
            Readmission Risk
          </span>
        </div>
      </div>

      <div style={{ textAlign: 'center' }}>
        <span className={`badge ${bgClass}`} style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
          <Icon size={16} />
          {category}
        </span>
      </div>

      {/* Clinical Recommendation Box */}
      <div style={{
        padding: '12px 16px',
        borderRadius: '10px',
        background: category.includes('High') || category.includes('Critical') ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
        borderLeft: `4px solid ${gaugeColor}`,
        border: `1px solid ${category.includes('High') || category.includes('Critical') ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`
      }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: gaugeColor, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
          Clinical Protocol Recommendation
        </div>
        <div style={{ fontSize: '0.85rem', color: '#E2E8F0', lineHeight: '1.4' }}>
          {recommendation || 'Standard post-ICU step-down monitoring. Re-evaluate vital signs prior to discharge.'}
        </div>
      </div>
    </div>
  );
}
