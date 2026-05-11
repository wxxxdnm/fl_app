import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, List, Timeline, Button, Space, Spin, message, Tag, Popconfirm } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { DatabaseOutlined, TeamOutlined, LineChartOutlined, PlayCircleOutlined, SettingOutlined, LaptopOutlined, DashboardOutlined, AppstoreOutlined, RightOutlined, SyncOutlined, DeleteOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const DashboardContainer = styled.div`
  padding: 0;
`;

const QuickActionCard = styled(Card)`
  height: 100%;
  cursor: pointer;
  border-radius: 28px;
  transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;

  &:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: rgba(0, 113, 227, 0.18);
    box-shadow: var(--app-shadow);
  }
`;

const HeroSection = styled.div`
  position: relative;
  overflow: hidden;
  margin-bottom: 28px;
  padding: 40px 46px;
  border: 1px solid rgba(255, 255, 255, 0.66);
  border-radius: 36px;
  background:
    radial-gradient(circle at 80% 20%, rgba(0, 113, 227, 0.16), transparent 26%),
    radial-gradient(circle at 90% 84%, rgba(142, 92, 247, 0.14), transparent 30%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.58));
  box-shadow: var(--app-shadow);
  backdrop-filter: blur(28px);

  &::after {
    content: '';
    position: absolute;
    right: -70px;
    top: -90px;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(0, 113, 227, 0.18), rgba(142, 92, 247, 0.16));
    filter: blur(4px);
  }

  @media (max-width: 768px) {
    padding: 30px 24px;
  }
`;

const HeroContent = styled.div`
  position: relative;
  z-index: 1;
  max-width: 620px;
`;

const Eyebrow = styled.div`
  margin-bottom: 12px;
  color: var(--app-blue);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
`;

const HeroText = styled.p`
  max-width: 540px;
  margin: 0 0 22px;
  color: var(--app-muted);
  font-size: 16px;
  line-height: 1.65;
`;

const SectionTitle = styled.h2`
  margin: 0 0 16px;
  font-size: 24px;
  font-weight: 800;
`;

const StatCard = styled(Card)`
  height: 100%;
  background: ${props => props.$tone};
`;

const IconBubble = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 58px;
  height: 58px;
  margin-bottom: 18px;
  border-radius: 20px;
  color: ${props => props.$color};
  font-size: 30px;
  background: ${props => props.$background};
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
    training_runs: [],
    saved_models: [],
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
  }, []);

  const deleteTrainingRun = async (runId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/train/history/${encodeURIComponent(runId)}`, {
        method: 'DELETE'
      });
      const data = await response.json();
      if (response.ok) {
        message.success('历史训练记录已删除');
        fetchDashboardStats();
      } else {
        message.error(`删除失败: ${data.error}`);
      }
    } catch (error) {
      message.error('删除历史训练记录失败');
    }
  };

  const quickActions = [
    {
      title: '数据管理',
      description: '上传和管理数据集',
      icon: <DatabaseOutlined />,
      color: '#0071e3',
      background: 'rgba(0, 113, 227, 0.10)',
      path: '/data'
    },
    {
      title: '模型训练',
      description: '启动联邦学习训练',
      icon: <AppstoreOutlined />,
      color: '#30d158',
      background: 'rgba(48, 209, 88, 0.12)',
      path: '/train'
    },
    {
      title: '客户端管理',
      description: '监控客户端状态',
      icon: <TeamOutlined />,
      color: '#ff9f0a',
      background: 'rgba(255, 159, 10, 0.12)',
      path: '/clients'
    },
    {
      title: '可视化分析',
      description: '查看训练结果',
      icon: <LineChartOutlined />,
      color: '#8e5cf7',
      background: 'rgba(142, 92, 247, 0.12)',
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
      <HeroSection>
        <HeroContent>
          <Eyebrow>Federated Learning Studio</Eyebrow>
          <HeroText>
            统一管理数据、客户端和历史记录，快速查看性能、分布与训练进展。
          </HeroText>
          <Space wrap>
            <Button icon={<SyncOutlined />} onClick={fetchDashboardStats}>
              刷新数据
            </Button>
            <Button type="primary" size="large" icon={<PlayCircleOutlined />} onClick={() => navigate('/train')}>
              快速开始训练
            </Button>
          </Space>
        </HeroContent>
      </HeroSection>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <StatCard bordered={false} $tone="linear-gradient(135deg, rgba(232, 242, 255, 0.92), rgba(255, 255, 255, 0.76))">
            <Statistic
              title="总客户端数"
              value={stats.total_clients}
              prefix={<TeamOutlined />}
              valueStyle={{ color: 'var(--app-blue)' }}
            />
          </StatCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard bordered={false} $tone="linear-gradient(135deg, rgba(48, 209, 88, 0.13), rgba(255, 255, 255, 0.78))">
            <Statistic
              title="模型准确率"
              value={stats.latest_accuracy}
              precision={1}
              suffix="%"
              prefix={<DashboardOutlined />}
              valueStyle={{ color: 'var(--app-green)' }}
            />
          </StatCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard bordered={false} $tone="linear-gradient(135deg, rgba(255, 159, 10, 0.14), rgba(255, 255, 255, 0.78))">
            <Statistic
              title="训练轮次"
              value={stats.total_rounds}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: 'var(--app-orange)' }}
            />
          </StatCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard bordered={false} $tone="linear-gradient(135deg, rgba(142, 92, 247, 0.13), rgba(255, 255, 255, 0.78))">
            <Statistic
              title="支持数据集"
              value={stats.num_datasets}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: 'var(--app-purple)' }}
            />
          </StatCard>
        </Col>
      </Row>

      {/* 快速操作 */}
      <SectionTitle>功能模块</SectionTitle>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {quickActions.map((action, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <QuickActionCard
              onClick={() => navigate(action.path)}
              bodyStyle={{ minHeight: 218, textAlign: 'left', padding: '28px 24px' }}
            >
              <IconBubble $color={action.color} $background={action.background}>{action.icon}</IconBubble>
              <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 20, fontWeight: 800 }}>{action.title}</h3>
              <p style={{ color: 'var(--app-muted)', fontSize: '14px', lineHeight: 1.7, marginBottom: 18 }}>{action.description}</p>
              <Button type="link" style={{ padding: 0, color: action.color }} icon={<RightOutlined />}>进入模块</Button>
            </QuickActionCard>
          </Col>
        ))}
      </Row>

      {/* 训练进度和最近活动 */}
      <Row gutter={16}>
        <Col xs={24} lg={16}>
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
          <Card title="历史训练记录">
            <List
              dataSource={stats.training_runs || []}
              locale={{ emptyText: '暂无历史训练记录' }}
              renderItem={(run) => (
                <List.Item>
                  <List.Item.Meta
                    title={`${(run.dataset_name || '').toUpperCase()} / ${run.model_name || 'model'} / ${run.aggregation_algorithm || 'fedavg'}`}
                    description={`${new Date(run.timestamp).toLocaleString()} · ${run.rounds || 0} 轮 · ${run.num_clients || 0} 客户端 · 准确率 ${((run.final_accuracy || 0) * 100).toFixed(2)}%`}
                  />
                  <Tag color={run.status === 'Completed' ? 'green' : run.status === 'Error' ? 'red' : 'blue'}>{run.status}</Tag>
                  <Popconfirm
                    title="确认删除这条历史训练记录？"
                    okText="删除"
                    cancelText="取消"
                    onConfirm={() => deleteTrainingRun(run.id)}
                  >
                    <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
                  </Popconfirm>
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="历史模型" style={{ marginBottom: 16 }}>
            <List
              dataSource={stats.saved_models || []}
              locale={{ emptyText: '暂无历史模型' }}
              renderItem={(model) => (
                <List.Item>
                  <List.Item.Meta
                    title={model.filename}
                    description={`${(model.dataset_name || 'unknown').toUpperCase()} / ${model.model_name || model.model_class || 'model'}${model.rounds ? ` · ${model.rounds} 轮` : ''}`}
                  />
                </List.Item>
              )}
            />
          </Card>
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