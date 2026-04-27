import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/lib/locale/zh_CN';
import 'antd/dist/antd.css';
import styled from 'styled-components';
import DataManagement from './pages/DataManagement';
import ModelTraining from './pages/ModelTraining';
import ClientManagement from './pages/ClientManagement';
import Visualization from './pages/Visualization';
import Home from './pages/Home';
import Header from './components/Header';

const AppContainer = styled.div`
  min-height: 100vh;
  background-color: #f0f2f5;
`;

const MainContent = styled.div`
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
`;

function App() {
  return (
    <ConfigProvider locale={zhCN}>
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