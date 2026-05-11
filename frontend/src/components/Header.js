import React from 'react';
import { Layout, Menu } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import { HomeOutlined, DatabaseOutlined, PlayCircleOutlined, TeamOutlined, LineChartOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Header: AntHeader } = Layout;

const StyledHeader = styled(AntHeader)`
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  height: 72px;
  padding: 0 max(24px, calc((100vw - 1280px) / 2 + 24px));
  border-bottom: 1px solid rgba(29, 29, 31, 0.08);
  background: rgba(251, 251, 253, 0.76);
  box-shadow: 0 8px 30px rgba(31, 35, 41, 0.06);
  backdrop-filter: saturate(180%) blur(24px);

  .ant-menu {
    flex: 1;
    min-width: 0;
    background: transparent;
  }

  .ant-menu-horizontal {
    line-height: 72px;
  }

  .ant-menu-horizontal > .ant-menu-item {
    top: 0;
    margin: 0 3px;
    padding: 0 15px;
    border-bottom: none;
    border-radius: 999px;
    color: var(--app-muted);
    font-weight: 600;
    transition: background 0.2s ease, color 0.2s ease;
  }

  .ant-menu-horizontal > .ant-menu-item:hover,
  .ant-menu-horizontal > .ant-menu-item-active,
  .ant-menu-horizontal > .ant-menu-item-selected {
    border-bottom: none;
    background: rgba(0, 113, 227, 0.08);
    color: var(--app-blue);
  }

  .ant-menu-horizontal > .ant-menu-item::after {
    display: none;
  }

  @media (max-width: 920px) {
    height: auto;
    min-height: 72px;
    flex-wrap: wrap;
    padding: 12px 16px;

    .ant-menu-horizontal {
      width: 100%;
      line-height: 46px;
      overflow-x: auto;
      white-space: nowrap;
    }
  }
`;

const Logo = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-right: 40px;
  color: var(--app-text);
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.04em;
  white-space: nowrap;

  &::before {
    content: '';
    width: 34px;
    height: 34px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0071e3 0%, #8e5cf7 100%);
    box-shadow: 0 12px 26px rgba(0, 113, 227, 0.24);
  }
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
        style={{ borderBottom: 'none' }}
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