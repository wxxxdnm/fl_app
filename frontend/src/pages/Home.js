import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, List, Timeline, Button, Space, Spin, message } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { DatabaseOutlined, TeamOutlined, LineChartOutlined, PlayCircleOutlined, SettingOutlined, LaptopOutlined, DashboardOutlined, AppstoreOutlined, RightOutlined, SyncOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const DashboardContainer = styled.div`
  padding: 20px;
`;

const QuickActionCard = styled(Card)`
  cursor: pointer;
  transition: all 0.3s;
  border-radius: 8px;
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border-color: #1890ff;
  }
`;

const Home = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total_clients: 0,
    latest_accuracy: 0,
    total_rounds: 0,
    num_datasets: 0,
    training_history: [],
    activities: []
  });

  const fetchDashboardStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/main/dashboard_stats');
      const data = await response.json();
      if (response.ok) {
        setStats(data);
      } else {
        message.error('获取主页统计数据失败');
      }
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      message.error('获取主页统计数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardStats();
    
    // 每 10 秒刷新一次数据，以保持系统动态最新
    const interval = setInterval(fetchDashboardStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const quickActions = [
    {
      title: '数据管理',
      description: '上传和管理数据集',
      icon: <DatabaseOutlined style={{ fontSize: '32px', color: '#1890ff' }} />,
      path: '/data'
    },
    {
      title: '模型训练',
      description: '启动联邦学习训练',
      icon: <AppstoreOutlined style={{ fontSize: '32px', color: '#52c41a' }} />,
      path: '/train'
    },
    {
      title: '客户端管理',
      description: '监控客户端状态',
      icon: <TeamOutlined style={{ fontSize: '32px', color: '#fa8c16' }} />,
      path: '/clients'
    },
    {
      title: '可视化分析',
      description: '查看训练结果',
      icon: <LineChartOutlined style={{ fontSize: '32px', color: '#722ed1' }} />,
      path: '/visualization'
    }
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px' }}>
        <Spin size="large" tip="正在加载数据..." />
      </div>
    );
  }

  return (
    <DashboardContainer>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h1 style={{ margin: 0 }}>联邦学习平台概览</h1>
        <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => navigate('/train')}>
          快速开始训练
        </Button>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card bordered={false} style={{ background: '#e6f7ff' }}>
            <Statistic
              title="总客户端数"
              value={stats.total_clients}
              prefix={<TeamOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card bordered={false} style={{ background: '#f6ffed' }}>
            <Statistic
              title="模型准确率"
              value={stats.latest_accuracy}
              precision={1}
              suffix="%"
              prefix={<DashboardOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card bordered={false} style={{ background: '#fff7e6' }}>
            <Statistic
              title="训练轮次"
              value={stats.total_rounds}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card bordered={false} style={{ background: '#f9f0ff' }}>
            <Statistic
              title="支持数据集"
              value={stats.num_datasets}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <h2 style={{ marginBottom: 16 }}>功能模块</h2>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {quickActions.map((action, index) => (
          <Col span={6} key={index}>
            <QuickActionCard
              onClick={() => navigate(action.path)}
              bodyStyle={{ textAlign: 'center', padding: '24px 16px' }}
            >
              {action.icon}
              <h3 style={{ marginTop: 16, marginBottom: 8 }}>{action.title}</h3>
              <p style={{ color: '#666', fontSize: '14px', marginBottom: 16 }}>{action.description}</p>
              <Button type="link" icon={<RightOutlined />}>进入模块</Button>
            </QuickActionCard>
          </Col>
        ))}
      </Row>

      {/* 训练进度和最近活动 */}
      <Row gutter={16}>
        <Col span={16}>
          <Card title="最近训练趋势" style={{ marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={stats.training_history.length > 0 ? stats.training_history : [{round: 0, accuracy: 0, loss: 0}]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: '训练轮次', position: 'insideBottomRight', offset: -5 }} />
                <YAxis label={{ value: '数值', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#1890ff"
                  strokeWidth={3}
                  name="准确率"
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#ff4d4f"
                  strokeWidth={3}
                  name="损失"
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="系统动态">
            <Timeline mode="left">
              {stats.activities && stats.activities.length > 0 ? (
                stats.activities.map((activity, index) => (
                  <Timeline.Item 
                    key={activity.id} 
                    color={
                      activity.type === 'success' ? 'green' : 
                      activity.type === 'error' ? 'red' : 
                      activity.type === 'warning' ? 'orange' : 
                      activity.type === 'process' ? 'blue' : 'gray'
                    }
                  >
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <span>{activity.content}</span>
                      <small style={{ color: '#999' }}>{new Date(activity.timestamp).toLocaleString()}</small>
                    </div>
                  </Timeline.Item>
                ))
              ) : (
                <Timeline.Item color="gray">暂无系统动态</Timeline.Item>
              )}
            </Timeline>
          </Card>
        </Col>
      </Row>
    </DashboardContainer>
  );
};

export default Home;