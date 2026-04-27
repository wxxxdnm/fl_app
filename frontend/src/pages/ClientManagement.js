import React, { useState, useEffect } from 'react';
import { Card, Table, Tag, Statistic, Row, Col, Space, Button, Progress, Alert, Tabs, Divider } from 'antd';
import { TeamOutlined, MonitorOutlined, WarningOutlined, CheckCircleOutlined, SyncOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import styled from 'styled-components';

const PageContainer = styled.div`
  padding: 20px;
`;

const ClientCard = styled(Card)`
  margin-bottom: 16px;
`;

const ClientStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => {
    switch(props.status) {
      case 'active': return '#52c41a';
      case 'inactive': return '#ff4d4f';
      case 'busy': return '#fa8c16';
      default: return '#8c8c8c';
    }
  }};
`;

const ClientManagement = () => {
  const [clients, setClients] = useState([]);
  const [stats, setStats] = useState({
    total_clients: 0,
    active_clients: 0,
    busy_clients: 0,
    inactive_clients: 0
  });
  const [performance, setPerformance] = useState({
    participation_distribution: [],
    training_time_stats: [],
    system_resources: {
      cpu: 0,
      memory: 0,
      gpu: {
        available: false,
        usage: 0,
        memory: 0,
        memory_used_mb: 0,
        memory_total_mb: 0,
        name: 'No CUDA GPU'
      },
      gpu_usage: 0,
      gpu_memory: 0,
      latency: 0,
      bandwidth: 0
    }
  });
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedClient, setSelectedClient] = useState(null);

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = () => {
    fetchClientStats();
    fetchClientDetails();
    fetchClientPerformance();
  };

  const fetchClientStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/clients/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('获取客户端统计失败', error);
    }
  };

  const fetchClientDetails = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/clients/');
      const data = await response.json();
      // 将对象转换为数组供 Table 使用
      const clientArray = Object.values(data.clients || {});
      setClients(clientArray);
    } catch (error) {
      console.error('获取客户端详情失败', error);
    }
  };

  const fetchClientPerformance = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/clients/performance');
      const data = await response.json();
      if (response.ok) {
        setPerformance(data);
      } else {
        console.error('后端返回错误:', data.error);
      }
    } catch (error) {
      console.error('获取客户端性能数据失败', error);
    }
  };

  const getClientDetails = async (clientId) => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/clients/${clientId}/metrics`);
      const data = await response.json();
      setSelectedClient({ ...data, client_id: clientId });
    } catch (error) {
      console.error('获取客户端详情失败', error);
    } finally {
      setLoading(false);
    }
  };

  const updateClientStatus = async (clientId, status) => {
    try {
      await fetch(`http://localhost:5000/api/clients/${clientId}/status`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ status })
      });
      fetchClientDetails();
    } catch (error) {
      console.error('更新客户端状态失败', error);
    }
  };

  const clientColumns = [
    {
      title: '客户端ID',
      dataIndex: 'client_id',
      key: 'client_id',
      sorter: (a, b) => a.client_id - b.client_id,
      width: '10%'
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record) => (
        <ClientStatus>
          <StatusDot status={record.status} />
          <Tag color={
            record.status === 'active' ? 'green' :
            record.status === 'busy' ? 'orange' : 'red'
          }>
            {record.status === 'active' ? '活跃' :
             record.status === 'busy' ? '忙碌' : '离线'}
          </Tag>
        </ClientStatus>
      ),
      width: '15%'
    },
    {
      title: '计算能力',
      dataIndex: 'compute_power',
      key: 'compute_power',
      render: (power) => (
        <Tag color={
          power === 'high' ? 'green' :
          power === 'medium' ? 'orange' : 'red'
        }>
          {power === 'high' ? '高' :
           power === 'medium' ? '中' : '低'}
        </Tag>
      ),
      width: '12%'
    },
    {
      title: '网络状况',
      dataIndex: 'network_speed',
      key: 'network_speed',
      render: (speed) => (
        <Progress
          percent={speed === 'excellent' ? 100 : speed === 'good' ? 70 : 40}
          size="small"
          status={speed === 'excellent' ? 'success' : speed === 'good' ? 'normal' : 'exception'}
          showInfo={false}
        />
      ),
      width: '15%'
    },
    {
      title: '数据质量',
      dataIndex: 'data_quality',
      key: 'data_quality',
      render: (quality) => (
        <Progress
          percent={quality === 'high' ? 90 : quality === 'medium' ? 60 : 30}
          size="small"
          strokeColor={quality === 'high' ? '#52c41a' : quality === 'medium' ? '#fa8c16' : '#ff4d4f'}
          showInfo={false}
        />
      ),
      width: '15%'
    },
    {
      title: '参与次数',
      dataIndex: 'participation_count',
      key: 'participation_count',
      sorter: (a, b) => a.participation_count - b.participation_count,
      width: '12%'
    },
    {
      title: '平均训练时间',
      dataIndex: 'avg_training_time',
      key: 'avg_training_time',
      render: (time) => `${time}s`,
      width: '12%'
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            onClick={() => getClientDetails(record.client_id)}
          >
            详情
          </Button>
          {record.status === 'active' && (
            <Button
              type="link"
              size="small"
              danger
              onClick={() => updateClientStatus(record.client_id, 'inactive')}
            >
              停用
            </Button>
          )}
        </Space>
      ),
      width: '10%'
    }
  ];

  return (
    <PageContainer>
      <h1>客户端管理</h1>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <Tabs.TabPane tab={<span><TeamOutlined />客户端概览</span>} key="overview">
          {/* 统计卡片 */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <ClientCard>
                <Statistic
                  title="活跃客户端"
                  value={stats.active_clients}
                  valueStyle={{ color: '#52c41a' }}
                  prefix={<CheckCircleOutlined />}
                />
                <Progress
                  percent={stats.total_clients > 0 ? (stats.active_clients / stats.total_clients) * 100 : 0}
                  showInfo={false}
                  strokeColor="#52c41a"
                  style={{ marginTop: 8 }}
                />
              </ClientCard>
            </Col>
            <Col span={8}>
              <ClientCard>
                <Statistic
                  title="忙碌客户端"
                  value={stats.busy_clients}
                  valueStyle={{ color: '#fa8c16' }}
                  prefix={<SyncOutlined />}
                />
                <Progress
                  percent={stats.total_clients > 0 ? (stats.busy_clients / stats.total_clients) * 100 : 0}
                  showInfo={false}
                  strokeColor="#fa8c16"
                  style={{ marginTop: 8 }}
                />
              </ClientCard>
            </Col>
            <Col span={8}>
              <ClientCard>
                <Statistic
                  title="离线客户端"
                  value={stats.inactive_clients}
                  valueStyle={{ color: '#ff4d4f' }}
                  prefix={<WarningOutlined />}
                />
                <Progress
                  percent={stats.total_clients > 0 ? (stats.inactive_clients / stats.total_clients) * 100 : 0}
                  showInfo={false}
                  strokeColor="#ff4d4f"
                  style={{ marginTop: 8 }}
                />
              </ClientCard>
            </Col>
          </Row>

          {/* 客户端列表 */}
          <ClientCard title="客户端列表">
            <Table
              columns={clientColumns}
              dataSource={clients}
              rowKey="client_id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true
              }}
              scroll={{ x: 1200 }}
            />
          </ClientCard>
        </Tabs.TabPane>

        <Tabs.TabPane tab={<span><MonitorOutlined />性能监控</span>} key="monitoring">
          <Row gutter={16}>
            <Col span={12}>
              <ClientCard title="客户端参与度分布">
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={performance.participation_distribution || []}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#1890ff">
                        {(performance.participation_distribution || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={['#1890ff', '#52c41a', '#fa8c16', '#722ed1', '#eb2f96'][index % 5]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </ClientCard>
            </Col>
            <Col span={12}>
              <ClientCard title="平均训练时间 (秒)">
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={performance.training_time_stats || []}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#52c41a" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </ClientCard>
            </Col>
          </Row>

          <ClientCard title="系统资源使用情况" style={{ marginTop: 16 }}>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic
                  title="平均CPU使用率"
                  value={performance.system_resources?.cpu || 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#1890ff' }}
                />
                <Progress 
                  percent={performance.system_resources?.cpu || 0} 
                  showInfo={false} 
                  size="small" 
                  strokeColor="#1890ff"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="平均内存使用率"
                  value={performance.system_resources?.memory || 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
                <Progress 
                  percent={performance.system_resources?.memory || 0} 
                  showInfo={false} 
                  size="small" 
                  strokeColor="#52c41a"
                />
              </Col>
              <Col span={8}>
                <Statistic
                title="GPU使用率"
                value={performance.system_resources?.gpu?.usage || performance.system_resources?.gpu_usage || 0}
                precision={1}
                suffix="%"
                valueStyle={{ color: performance.system_resources?.gpu?.available ? '#13c2c2' : '#8c8c8c' }}
              />
              <Progress
                percent={performance.system_resources?.gpu?.usage || performance.system_resources?.gpu_usage || 0}
                showInfo={false}
                size="small"
                strokeColor={performance.system_resources?.gpu?.available ? '#13c2c2' : '#d9d9d9'}
              />
                <div style={{ marginTop: 4, color: '#8c8c8c', fontSize: 12 }}>
                  {performance.system_resources?.gpu?.available
                    ? `${performance.system_resources.gpu.name}：${performance.system_resources.gpu.memory_used_mb}/${performance.system_resources.gpu.memory_total_mb} MB`
                    : '未检测到 CUDA GPU'}
                </div>
              </Col>
              <Col span={8}>
                <Statistic
                  title="平均网络延迟"
                  value={performance.system_resources?.latency || 0}
                  precision={0}
                  suffix="ms"
                  valueStyle={{ color: '#fa8c16' }}
                />
                <Progress 
                  percent={Math.min(100, ((performance.system_resources?.latency || 0) / 200) * 100)} 
                  showInfo={false} 
                  size="small" 
                  strokeColor="#fa8c16"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="平均带宽"
                  value={performance.system_resources?.bandwidth || 0}
                  precision={1}
                  suffix="Mbps"
                  valueStyle={{ color: '#722ed1' }}
                />
                <Progress 
                  percent={Math.min(100, ((performance.system_resources?.bandwidth || 0) / 200) * 100)} 
                  showInfo={false} 
                  size="small" 
                  strokeColor="#722ed1"
                />
              </Col>
            </Row>
          </ClientCard>
        </Tabs.TabPane>

        {selectedClient && (
          <Tabs.TabPane tab={<span><InfoCircleOutlined />客户端详情</span>} key="details">
            <ClientCard title={`客户端 ${selectedClient.client_id} 详情`}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="状态"
                    value={selectedClient.status}
                    valueStyle={{ color: selectedClient.status === 'active' ? '#52c41a' : '#ff4d4f' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="计算能力"
                    value={selectedClient.compute_power}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="参与次数"
                    value={selectedClient.participation_count}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Col>
              </Row>
              <Divider />
              <p><strong>网络状况：</strong>{selectedClient.network_speed}</p>
              <p><strong>数据质量：</strong>{selectedClient.data_quality}</p>
              <p><strong>平均训练时间：</strong>{selectedClient.avg_training_time}s</p>
              <p><strong>最后活动：</strong>{new Date(selectedClient.last_activity).toLocaleString()}</p>
            </ClientCard>
          </Tabs.TabPane>
        )}
      </Tabs>
    </PageContainer>
  );
};

export default ClientManagement;
