import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/lib/locale/zh_CN';
import 'antd/dist/antd.css';
import styled, { createGlobalStyle } from 'styled-components';
import DataManagement from './pages/DataManagement';
import ModelTraining from './pages/ModelTraining';
import ClientManagement from './pages/ClientManagement';
import Visualization from './pages/Visualization';
import Home from './pages/Home';
import Header from './components/Header';

const GlobalStyle = createGlobalStyle`
  :root {
    --app-blue: #0071e3;
    --app-blue-soft: #e8f2ff;
    --app-green: #30d158;
    --app-orange: #ff9f0a;
    --app-purple: #8e5cf7;
    --app-red: #ff453a;
    --app-text: #1d1d1f;
    --app-muted: #6e6e73;
    --app-border: rgba(29, 29, 31, 0.08);
    --app-glass: rgba(255, 255, 255, 0.72);
    --app-shadow: 0 24px 70px rgba(31, 35, 41, 0.10);
    --app-shadow-soft: 0 12px 34px rgba(31, 35, 41, 0.08);
  }

  * {
    box-sizing: border-box;
  }

  html {
    background: #f5f5f7;
  }

  body {
    margin: 0;
    color: var(--app-text);
    background: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
  }

  h1, h2, h3, h4, h5 {
    color: var(--app-text);
    letter-spacing: -0.03em;
  }

  .ant-card {
    overflow: hidden;
    border: 1px solid var(--app-border);
    border-radius: 24px;
    background: var(--app-glass);
    box-shadow: var(--app-shadow-soft);
    backdrop-filter: blur(24px);
  }

  .ant-card-head {
    min-height: 58px;
    border-bottom: 1px solid rgba(29, 29, 31, 0.06);
    color: var(--app-text);
    font-weight: 700;
  }

  .ant-card-head-title {
    letter-spacing: -0.02em;
  }

  .ant-card-body {
    color: var(--app-text);
  }

  .ant-btn {
    height: 38px;
    border-radius: 999px;
    border-color: rgba(29, 29, 31, 0.10);
    box-shadow: none;
    font-weight: 600;
    transition: transform 0.22s ease, box-shadow 0.22s ease, background 0.22s ease, border-color 0.22s ease;
  }

  .ant-btn:hover,
  .ant-btn:focus {
    transform: translateY(-1px);
    border-color: rgba(0, 113, 227, 0.32);
    box-shadow: 0 10px 24px rgba(0, 113, 227, 0.12);
  }

  .ant-btn-primary {
    border: none;
    background: linear-gradient(135deg, #0071e3 0%, #5e9dff 100%);
    box-shadow: 0 14px 30px rgba(0, 113, 227, 0.24);
  }

  .ant-btn-primary:hover,
  .ant-btn-primary:focus {
    background: linear-gradient(135deg, #0066cc 0%, #438cff 100%);
  }

  .ant-btn-dangerous {
    border-color: rgba(255, 69, 58, 0.25);
  }

  .ant-input,
  .ant-input-number,
  .ant-select-selector {
    border-radius: 14px !important;
    border-color: rgba(29, 29, 31, 0.10) !important;
    background: rgba(255, 255, 255, 0.76) !important;
  }

  .ant-select-dropdown,
  .ant-picker-dropdown,
  .ant-dropdown-menu {
    overflow: hidden;
    border-radius: 18px;
    box-shadow: 0 22px 58px rgba(31, 35, 41, 0.16);
  }

  .ant-table {
    border-radius: 18px;
    background: transparent;
  }

  .ant-table-thead > tr > th {
    border-bottom: 1px solid rgba(29, 29, 31, 0.06);
    background: rgba(245, 245, 247, 0.72);
    color: var(--app-muted);
    font-weight: 700;
  }

  .ant-table-tbody > tr > td {
    border-bottom: 1px solid rgba(29, 29, 31, 0.05);
  }

  .ant-table-tbody > tr:hover > td {
    background: rgba(0, 113, 227, 0.045) !important;
  }

  .ant-tag {
    border-radius: 999px;
    padding: 2px 10px;
    border: none;
    font-weight: 600;
  }

  .ant-tabs-tab {
    border-radius: 999px;
    padding: 10px 18px;
  }

  .ant-tabs-tab.ant-tabs-tab-active {
    background: rgba(0, 113, 227, 0.08);
  }

  .ant-tabs-ink-bar {
    display: none;
  }

  .ant-tabs-top > .ant-tabs-nav::before {
    border-bottom: none;
  }

  .ant-statistic-title {
    color: var(--app-muted);
    font-weight: 600;
  }

  .ant-statistic-content {
    color: var(--app-text);
    letter-spacing: -0.04em;
  }

  .recharts-cartesian-grid line {
    stroke: rgba(29, 29, 31, 0.06);
  }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  position: relative;
  overflow-x: hidden;
  background:
    radial-gradient(circle at 12% 8%, rgba(0, 113, 227, 0.16), transparent 28%),
    radial-gradient(circle at 82% 12%, rgba(142, 92, 247, 0.14), transparent 30%),
    linear-gradient(180deg, #fbfbfd 0%, #f5f5f7 42%, #eef2f8 100%);

  &::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: linear-gradient(rgba(255, 255, 255, 0.32) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.32) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.38), transparent 62%);
  }
`;

const MainContent = styled.div`
  position: relative;
  z-index: 1;
  padding: 32px 24px 56px;
  max-width: 1280px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: 20px 14px 40px;
  }
`;

function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <GlobalStyle />
      <Router>
        <AppContainer>
          <Header />
          <MainContent>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/train" element={<ModelTraining />} />
              <Route path="/clients" element={<ClientManagement />} />
              <Route path="/visualization" element={<Visualization />} />
            </Routes>
          </MainContent>
        </AppContainer>
      </Router>
    </ConfigProvider>
  );
}

export default App;