import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Database, HardDrive, Cpu, ArrowRightLeft, Lock, FileText, Activity } from 'lucide-react';

export default function ComparisonTool() {
  const [numHospitals, setNumHospitals] = useState(5);
  const [patientsPerHospital, setPatientsPerHospital] = useState(2500);

  const totalPatients = numHospitals * patientsPerHospital;
  // Raw patient record size estimate ~500 KB per FHIR bundle
  const rawDataPerPatientMB = 0.5;
  const totalRawDataGB = (totalPatients * rawDataPerPatientMB) / 1024;
  
  // Model update weight vector size ~ 180 KB
  const modelWeightKB = 180;
  const totalModelWeightMB = (numHospitals * modelWeightKB) / 1024;

  const dataSavedPct = 99.98;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Header Banner */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#F8FAFC', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <ArrowRightLeft size={22} color="#38BDF8" /> Traditional Centralized ML vs. FedRisk Decentralized Architecture
        </h2>
        <p style={{ fontSize: '0.88rem', color: '#94A3B8', lineHeight: '1.6' }}>
          Demonstrates how traditional central AI systems require transmitting sensitive raw patient EHR records over public networks, creating massive data breach risks and violating privacy regulations like India's <strong>DPDP Act 2023</strong> and <strong>HIPAA</strong>. <strong>FedRisk</strong> executes GNN training locally inside hospital firewalls and transmits only zero-sum encrypted model weights.
        </p>
      </div>

      {/* Interactive Simulation Controls */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '16px' }}>
          Interactive Network Impact & Privacy Simulator
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '24px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600, display: 'block', marginBottom: '8px' }}>
              Participating Hospital Nodes: <strong style={{ color: '#F8FAFC' }}>{numHospitals} Hospitals</strong>
            </label>
            <input
              type="range"
              min="3"
              max="20"
              value={numHospitals}
              onChange={e => setNumHospitals(parseInt(e.target.value))}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600, display: 'block', marginBottom: '8px' }}>
              Patients per Hospital Cohort: <strong style={{ color: '#F8FAFC' }}>{patientsPerHospital.toLocaleString()} Patients</strong>
            </label>
            <input
              type="range"
              min="500"
              max="10000"
              step="500"
              value={patientsPerHospital}
              onChange={e => setPatientsPerHospital(parseInt(e.target.value))}
            />
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '10px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Total Network Cohort</span>
            <span style={{ fontSize: '1.4rem', fontWeight: 800, color: '#38BDF8', fontFamily: 'Outfit' }}>
              {totalPatients.toLocaleString()} ICU Trajectories
            </span>
          </div>
        </div>
      </div>

      {/* Side-by-Side Architectural Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '24px' }}>
        
        {/* Centralized Model Card */}
        <div className="glass-panel" style={{ padding: '24px', border: '1px solid rgba(239, 68, 68, 0.3)', background: 'rgba(239, 68, 68, 0.03)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <ShieldAlert size={24} color="#EF4444" />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F87171' }}>Traditional Centralized AI</h3>
            </div>
            <span className="badge badge-danger">High Risk</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Raw Patient Data Transferred</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#EF4444', fontFamily: 'Outfit' }}>
                {totalRawDataGB.toFixed(2)} GB
              </div>
              <div style={{ fontSize: '0.75rem', color: '#F87171', marginTop: '2px' }}>
                Full PHI / FHIR Bundles pooled centrally
              </div>
            </div>

            <div style={{ fontSize: '0.85rem', color: '#E2E8F0', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#EF4444', fontWeight: 700 }}>❌</span>
                <span><strong>DPDP Act 2023 & HIPAA Non-Compliant</strong>: Pooling raw medical records violates cross-border data residency & consent mandates.</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#EF4444', fontWeight: 700 }}>❌</span>
                <span><strong>Single Point of Vulnerability</strong>: Centralized database is a prime target for ransomware & data leakage.</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#EF4444', fontWeight: 700 }}>❌</span>
                <span><strong>High Network Overhead</strong>: Requires transmitting gigabytes of tabular FHIR JSON & medical records.</span>
              </div>
            </div>
          </div>
        </div>

        {/* FedRisk Decentralized Card */}
        <div className="glass-panel" style={{ padding: '24px', border: '1px solid rgba(16, 185, 129, 0.3)', background: 'rgba(16, 185, 129, 0.03)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <ShieldCheck size={24} color="#10B981" />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#34D399' }}>FedRisk Decentralized GNN</h3>
            </div>
            <span className="badge badge-success">Fully Compliant</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Raw Patient Data Transferred</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#34D399', fontFamily: 'Outfit' }}>
                0.00 Bytes <span style={{ fontSize: '0.85rem', color: '#94A3B8' }}>({totalModelWeightMB.toFixed(2)} MB Encrypted Weights)</span>
              </div>
              <div style={{ fontSize: '0.75rem', color: '#34D399', marginTop: '2px' }}>
                100% Data Confidentiality inside hospital firewall
              </div>
            </div>

            <div style={{ fontSize: '0.85rem', color: '#E2E8F0', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#10B981', fontWeight: 700 }}>✅</span>
                <span><strong>DPDP Act 2023 & HIPAA Compliant</strong>: Zero raw patient data leaves the local hospital infrastructure.</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#10B981', fontWeight: 700 }}>✅</span>
                <span><strong>Zero-Knowledge SMPC Defense</strong>: Local updates are blinded using zero-sum secret sharing, preventing weight inversion.</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{ color: '#10B981', fontWeight: 700 }}>✅</span>
                <span><strong>99.9% Bandwidth Reduction</strong>: Only small model gradient vectors are transmitted during FedAvg rounds.</span>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Metric Breakdown Table */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '14px' }}>
          Detailed Architecture Comparison Matrix
        </h3>

        <table className="custom-table">
          <thead>
            <tr>
              <th>Evaluation Metric</th>
              <th>Centralized Model</th>
              <th>FedRisk Platform</th>
              <th>Advantage / Impact</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ fontWeight: 600, color: '#F8FAFC' }}>Patient Data Ingestion</td>
              <td style={{ color: '#F87171' }}>Centralized Database Upload</td>
              <td style={{ color: '#34D399', fontWeight: 600 }}>Local Native FHIR Node Processing</td>
              <td>Zero Raw File Movement</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 600, color: '#F8FAFC' }}>Regulatory Compliance</td>
              <td style={{ color: '#F87171' }}>Non-Compliant (Exposure Risk)</td>
              <td style={{ color: '#34D399', fontWeight: 600 }}>DPDP Act 2023 & HIPAA Compliant</td>
              <td>Full Legal Safety</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 600, color: '#F8FAFC' }}>Network Bandwidth ({totalPatients.toLocaleString()} Pts)</td>
              <td style={{ color: '#F87171' }}>{totalRawDataGB.toFixed(2)} GB Transferred</td>
              <td style={{ color: '#38BDF8', fontWeight: 600 }}>{totalModelWeightMB.toFixed(2)} MB Encrypted Parameters</td>
              <td>&gt;99.9% Bandwidth Reduction</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 600, color: '#F8FAFC' }}>Security Layer</td>
              <td style={{ color: '#F87171' }}>Standard TLS Transit Only</td>
              <td style={{ color: '#C084FC', fontWeight: 600 }}>SMPC Zero-Sum Pairwise Masking</td>
              <td>Mathematical Gradient Blinding</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 600, color: '#F8FAFC' }}>Model Representation</td>
              <td style={{ color: '#94A3B8' }}>Tabular Regressors / Baseline ML</td>
              <td style={{ color: '#34D399', fontWeight: 600 }}>Temporal Patient Graph Neural Network (PyG)</td>
              <td>Temporal Disease Succession</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  );
}
