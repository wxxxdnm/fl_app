import React from 'react';
import { Layout, Menu } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import { HomeOutlined, DatabaseOutlined, PlayCircleOutlined, TeamOutlined, LineChartOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Header: AntHeader } = Layout;

const StyledHeader = styled(AntHeader)`
  background: #fff;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const Logo = styled.div`
  font-size: 20px;
  font-weight: bold;
  color: #1890ff;
  display: inline-block;
  margin-right: 48px;
`;

const Header = () => {
  const location = useLocation();
  const selectedKey = ['/', '/data', '/train', '/clients', '/visualization'].find(
    path => path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)
  ) || '/';

  return (
    <StyledHeader>
      <Logo>联邦学习平台</Logo>
      <Menu
        theme="light"
        mode="horizontal"
        selectedKeys={[selectedKey]}
        style={{ display: 'inline-block', borderBottom: 'none' }}
      >
        <Menu.Item key="/" icon={<HomeOutlined />}>
          <Link to="/">首页</Link>
        </Menu.Item>
        <Menu.Item key="/data" icon={<DatabaseOutlined />}>
          <Link to="/data">数据管理</Link>
        </Menu.Item>
        <Menu.Item key="/train" icon={<PlayCircleOutlined />}>
          <Link to="/train">模型训练</Link>
        </Menu.Item>
        <Menu.Item key="/clients" icon={<TeamOutlined />}>
          <Link to="/clients">客户端管理</Link>
        </Menu.Item>
        <Menu.Item key="/visualization" icon={<LineChartOutlined />}>
          <Link to="/visualization">可视化分析</Link>
        </Menu.Item>
      </Menu>
    </StyledHeader>
  );
};

export default Header;