import React, { useEffect, useState } from 'react';
import { BarChartOutlined, LineChartOutlined, PieChartOutlined, RadarChartOutlined } from '@ant-design/icons';
import { Button, Card, Col, Row, Select, Space, Tabs } from 'antd';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import styled from 'styled-components';

const PageContainer = styled.div`
  padding: 20px;
`;

const ChartCard = styled(Card)`
  margin-bottom: 16px;
`;

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d', '#ffc658', '#ff7c7c'];

const Visualization = () => {
  const [activeTab, setActiveTab] = useState('training');
  const [selectedDataset, setSelectedDataset] = useState('mnist');
  const [chartData, setChartData] = useState({
    trainingCurves: [],
    clientPerformance: [],
    confusionMatrix: { data: [], classes: [] },
    clientDistribution: [],
    distributionStats: []
  });

  useEffect(() => {
    loadVisualizationData();
  }, []);

  const loadVisualizationData = async () => {
    try {
      const trainingResponse = await fetch('http://localhost:5000/api/viz/training_curves');
      if (trainingResponse.ok) {
        const trainingData = await trainingResponse.json();
        setChartData(prev => ({ ...prev, trainingCurves: formatTrainingData(trainingData) }));
      }

      const performanceResponse = await fetch('http://localhost:5000/api/viz/model_performance');
      if (performanceResponse.ok) {
        const performanceData = await performanceResponse.json();
        setChartData(prev => ({ ...prev, clientPerformance: formatPerformanceData(performanceData) }));
      }

      const distributionResponse = await fetch('http://localhost:5000/api/viz/client_distribution');
      if (distributionResponse.ok) {
        const distributionData = await distributionResponse.json();
        setChartData(prev => ({
          ...prev,
          clientDistribution: formatDistributionData(distributionData),
          distributionStats: formatDistributionStats(distributionData.stats)
        }));
      }

      const confusionResponse = await fetch('http://localhost:5000/api/viz/confusion_matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_name: selectedDataset })
      });
      if (confusionResponse.ok) {
        const confusionData = await confusionResponse.json();
        setChartData(prev => ({ ...prev, confusionMatrix: formatConfusionMatrixData(confusionData) }));
      }
    } catch (error) {
      console.error('加载可视化数据失败', error);
    }
  };

  const formatTrainingData = (data) => {
    const rounds = data.rounds || [];
    const accuracies = data.global_accuracies || [];
    const losses = data.global_losses || [];

    return rounds.map((round, index) => ({
      round,
      accuracy: accuracies[index] || 0,
      loss: losses[index] || 0
    }));
  };

  const formatPerformanceData = (data) => {
    const clientIds = data.client_ids || [];
    const accuracies = data.accuracies || [];
    const losses = data.losses || [];

    return clientIds.map((id, index) => ({
      clientId: `客户端 ${id}`,
      accuracy: accuracies[index] || 0,
      loss: losses[index] || 0,
      samples: data.sample_sizes?.[index] || 0
    }));
  };

  const formatDistributionData = (data) => {
    const clientDistributions = data.client_distributions || [];
    const classNames = data.class_names || [];

    return clientDistributions.map(client => {
      const distribution = client.distribution || [];
      const counts = client.counts || [];
      return {
        clientId: `客户端 ${client.client_id}`,
        data: distribution.map((value, classIndex) => ({
          class: classNames[classIndex] || `类别 ${classIndex}`,
          value: value * 100,
          count: counts[classIndex] || 0
        }))
      };
    });
  };

  const formatDistributionStats = (stats = {}) => ([
    { subject: '类别均衡性', A: stats.class_balance || 0, fullMark: 100 },
    { subject: '数据质量', A: stats.data_quality || 0, fullMark: 100 },
    { subject: '样本数量', A: stats.sample_quantity || 0, fullMark: 100 },
    { subject: '特征多样性', A: stats.feature_diversity || 0, fullMark: 100 },
    { subject: '数据一致性', A: stats.data_consistency || 0, fullMark: 100 }
  ]);

  const formatConfusionMatrixData = (data) => {
    if (!data || !data.confusion_matrix || !data.class_names) return { data: [], classes: [] };

    const formatted = [];
    const matrix = data.confusion_matrix;
    const classes = data.class_names;

    for (let i = 0; i < matrix.length; i += 1) {
      for (let j = 0; j < matrix[i].length; j += 1) {
        formatted.push({
          actual: classes[i],
          predicted: classes[j],
          value: parseFloat((matrix[i][j] * 100).toFixed(1))
        });
      }
    }
    return { data: formatted, classes };
  };

  const renderConfusionMatrix = () => {
    const matrixInfo = chartData.confusionMatrix || { data: [], classes: [] };
    const matrixData = matrixInfo.data || [];
    const classes = matrixInfo.classes || [];

    if (matrixData.length === 0 || classes.length === 0) {
      return <p style={{ color: '#999' }}>暂无混淆矩阵数据</p>;
    }

    const numClasses = classes.length;
    const matrix = Array(numClasses).fill().map(() => Array(numClasses).fill(0));

    matrixData.forEach(item => {
      const actualIndex = classes.indexOf(item.actual);
      const predictedIndex = classes.indexOf(item.predicted);
      if (actualIndex >= 0 && predictedIndex >= 0) {
        matrix[actualIndex][predictedIndex] = item.value;
      }
    });

    return (
      <div style={{ textAlign: 'center', overflowX: 'auto' }}>
        <h3>混淆矩阵 (%)</h3>
        <div style={{ display: 'inline-block', marginTop: 20 }}>
          <table style={{ borderCollapse: 'collapse', border: '1px solid #ddd' }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid #ddd', padding: 8, backgroundColor: '#f5f5f5' }}>真实\预测</th>
                {classes.map(cls => (
                  <th key={cls} style={{ border: '1px solid #ddd', padding: 8, backgroundColor: '#f5f5f5' }}>{cls}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIndex) => (
                <tr key={classes[rowIndex]}>
                  <td style={{ border: '1px solid #ddd', padding: 8, fontWeight: 'bold', backgroundColor: '#f5f5f5' }}>{classes[rowIndex]}</td>
                  {row.map((value, columnIndex) => (
                    <td
                      key={`${classes[rowIndex]}-${classes[columnIndex]}`}
                      style={{
                        border: '1px solid #ddd',
                        padding: 8,
                        backgroundColor: rowIndex === columnIndex ? `rgba(82, 196, 26, ${value / 100})` : `rgba(255, 77, 79, ${value / 50})`,
                        color: value > 50 ? 'white' : 'black',
                        minWidth: 50
                      }}
                    >
                      {value}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <PageContainer>
      <h1>可视化分析</h1>

      <Space style={{ marginBottom: 16 }}>
        <span>选择数据集：</span>
        <Select
          value={selectedDataset}
          onChange={setSelectedDataset}
          style={{ width: 150 }}
        >
          <Select.Option value="mnist">MNIST</Select.Option>
          <Select.Option value="cifar10">CIFAR10</Select.Option>
        </Select>
        <Button onClick={loadVisualizationData}>刷新数据</Button>
      </Space>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <Tabs.TabPane
          tab={<span><LineChartOutlined />训练曲线</span>}
          key="training"
        >
          <Row gutter={16}>
            <Col span={12}>
              <ChartCard title="全局模型准确率">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData.trainingCurves}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#1890ff" strokeWidth={2} name="全局准确率" dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </Col>
            <Col span={12}>
              <ChartCard title="全局模型损失">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData.trainingCurves}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#ff4d4f" strokeWidth={2} name="全局损失" dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </Col>
          </Row>
        </Tabs.TabPane>

        <Tabs.TabPane
          tab={<span><BarChartOutlined />客户端性能</span>}
          key="performance"
        >
          <ChartCard title="客户端准确率对比">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData.clientPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={80} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="accuracy" fill="#52c41a" name="准确率" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="客户端样本数量分布">
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={chartData.clientPerformance}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ clientId, samples }) => `${clientId}: ${samples}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="samples"
                >
                  {chartData.clientPerformance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </Tabs.TabPane>

        <Tabs.TabPane
          tab={<span><PieChartOutlined />数据分布</span>}
          key="distribution"
        >
          <ChartCard title="客户端类别分布">
            {chartData.clientDistribution.map(client => (
              <div key={client.clientId} style={{ marginBottom: 24 }}>
                <h3>{client.clientId}</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={client.data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="class" />
                    <YAxis />
                    <Tooltip formatter={(value, name, item) => [`${value.toFixed(1)}% (${item.payload.count} 个样本)`, '数据占比']} />
                    <Bar dataKey="value" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </ChartCard>

          <ChartCard title="数据分布统计">
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData.distributionStats}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="数据分布评分" dataKey="A" stroke="#1890ff" fill="#1890ff" fillOpacity={0.6} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Tabs.TabPane>

        <Tabs.TabPane
          tab={<span><RadarChartOutlined />模型分析</span>}
          key="model"
        >
          <ChartCard title="混淆矩阵">
            {renderConfusionMatrix()}
          </ChartCard>
        </Tabs.TabPane>
      </Tabs>
    </PageContainer>
  );
};

export default Visualization;
