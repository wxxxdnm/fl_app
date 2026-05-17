import React, { useEffect, useState } from 'react';
import { BarChartOutlined, LineChartOutlined, PieChartOutlined, RadarChartOutlined } from '@ant-design/icons';
import { Button, Card, Col, Empty, Pagination, Row, Select, Space, Statistic, Table, Tabs, Tag } from 'antd';
import { useLocation } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
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
  padding: 0;
`;

const ChartCard = styled(Card)`
  margin-bottom: 20px;
  border-radius: 28px;
`;

const CLIENT_CHART_MIN_WIDTH = 900;
const CLIENT_BAR_WIDTH = 58;
const DISTRIBUTION_PAGE_SIZE = 6;

const CONTRIBUTION_WEIGHTS = {
  sample: 0.35,
  participation: 0.25,
  performance: 0.25,
  efficiency: 0.15
};

const Visualization = () => {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('training');
  const [selectedRunId, setSelectedRunId] = useState(location.state?.selectedRunId || null);
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [selectedRunSummary, setSelectedRunSummary] = useState(null);
  const [distributionPage, setDistributionPage] = useState(1);
  const [chartData, setChartData] = useState({
    trainingCurves: [],
    clientPerformance: [],
    clientContributions: [],
    confusionMatrix: { data: [], classes: [] },
    clientDistribution: [],
    distributionStats: []
  });

  useEffect(() => {
    loadVisualizationData(selectedRunId);
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    const selectedRun = trainingRuns.find(run => run.id === selectedRunId);
    if (selectedRun) {
      applyHistoricalRun(selectedRun);
    }
  }, [selectedRunId, trainingRuns]);

  const loadVisualizationData = async (targetRunId = selectedRunId) => {
    try {
      const dashboardResponse = await fetch('http://localhost:5000/api/main/dashboard_stats');
      if (dashboardResponse.ok) {
        const dashboardData = await dashboardResponse.json();
        const runs = dashboardData.training_runs || [];
        setTrainingRuns(runs);
        const selectedRun = runs.find(run => run.id === targetRunId) || runs[0];
        if (selectedRun) {
          setSelectedRunId(selectedRun.id);
          applyHistoricalRun(selectedRun);
          return;
        }
      }
      setSelectedRunId(null);
      setSelectedRunSummary(null);
      setChartData({
        trainingCurves: [],
        clientPerformance: [],
        clientContributions: [],
        confusionMatrix: { data: [], classes: [] },
        clientDistribution: [],
        distributionStats: []
      });
    } catch (error) {
      console.error('加载可视化数据失败', error);
    }
  };

  const handleRunChange = (runId) => {
    setSelectedRunId(runId);
    const selectedRun = trainingRuns.find(run => run.id === runId);
    if (selectedRun) {
      applyHistoricalRun(selectedRun);
    }
  };

  const formatIidLabel = (iid) => {
    if (iid === true) return 'IID';
    if (iid === false) return 'Non-IID';
    return '-';
  };

  const applyHistoricalRun = (run) => {
    const history = run.history || [];
    const finalRound = history[history.length - 1] || {};
    const visualization = run.visualization || {};
    const modelPerformance = visualization.model_performance || visualization.client_performance;
    const clientDistribution = visualization.client_distribution;
    const confusionMatrix = visualization.confusion_matrix;
    setDistributionPage(1);

    setChartData(prev => ({
      ...prev,
      trainingCurves: formatHistoricalTrainingData(history),
      clientPerformance: modelPerformance
        ? formatPerformanceData(modelPerformance)
        : formatHistoricalClientPerformance(finalRound),
      clientContributions: formatClientContributionData(history),
      clientDistribution: clientDistribution ? formatDistributionData(clientDistribution) : [],
      distributionStats: clientDistribution ? formatDistributionStats(clientDistribution.stats) : [],
      confusionMatrix: confusionMatrix ? formatConfusionMatrixData(confusionMatrix) : { data: [], classes: [] }
    }));
    setSelectedRunSummary({
      datasetName: run.dataset_name,
      modelName: run.model_name,
      algorithm: run.aggregation_algorithm,
      rounds: run.rounds,
      status: run.status,
      finalAccuracy: run.final_accuracy,
      finalLoss: run.final_loss,
      finalF1Score: run.final_f1_score,
      timestamp: run.timestamp,
      numClients: run.num_clients,
      iid: run.iid
    });
  };

  const formatHistoricalTrainingData = (history = []) => history.map(item => {
    const metrics = item.global_metrics || {};
    return {
      round: item.round,
      accuracy: metrics.accuracy || 0,
      loss: metrics.loss || 0,
      precision: metrics.precision || 0,
      recall: metrics.recall || 0,
      f1Score: metrics.f1_score || 0,
      balancedAccuracy: metrics.balanced_accuracy || 0,
      samplesPerSecond: metrics.samples_per_second || 0
    };
  });

  const formatHistoricalClientPerformance = (roundData = {}) => {
    const clientMetrics = roundData.client_metrics || [];

    return clientMetrics.map((metrics, index) => ({
      clientId: `客户端 ${metrics.client_id ?? index + 1}`,
      accuracy: metrics.accuracy || 0,
      loss: metrics.loss || 0,
      precision: metrics.precision || 0,
      recall: metrics.recall || 0,
      f1Score: metrics.f1_score || 0,
      balancedAccuracy: metrics.balanced_accuracy || 0,
      samplesPerSecond: metrics.samples_per_second || 0,
      samples: metrics.num_samples || 0
    }));
  };

  const formatPerformanceData = (data) => {
    const clientIds = data.client_ids || [];
    const accuracies = data.accuracies || [];
    const losses = data.losses || [];
    const precisions = data.precisions || [];
    const recalls = data.recalls || [];
    const f1Scores = data.f1_scores || [];
    const balancedAccuracies = data.balanced_accuracies || [];
    const samplesPerSecond = data.samples_per_second || [];

    return clientIds.map((id, index) => ({
      clientId: `客户端 ${id}`,
      accuracy: accuracies[index] || 0,
      loss: losses[index] || 0,
      precision: precisions[index] || 0,
      recall: recalls[index] || 0,
      f1Score: f1Scores[index] || 0,
      balancedAccuracy: balancedAccuracies[index] || 0,
      samplesPerSecond: samplesPerSecond[index] || 0,
      samples: data.sample_sizes?.[index] || 0
    }));
  };

  const getClientSortValue = (clientId) => {
    const match = String(clientId || '').match(/\d+/);
    return match ? Number(match[0]) : Number.MAX_SAFE_INTEGER;
  };

  const clampPercent = (value) => Math.max(0, Math.min(100, Number(value) || 0));

  const formatClientContributionData = (history = []) => {
    if (!history.length) return [];

    const accumulators = {};
    history.forEach(roundData => {
      (roundData.client_metrics || []).forEach(metrics => {
        const rawClientId = metrics.client_id ?? Object.keys(accumulators).length;
        const clientId = `客户端 ${rawClientId}`;
        if (!accumulators[clientId]) {
          accumulators[clientId] = {
            clientId,
            totalSamples: 0,
            participationRounds: 0,
            accuracySum: 0,
            f1Sum: 0,
            throughputSum: 0,
            trainingTimeSum: 0
          };
        }
        accumulators[clientId].totalSamples += Math.max(0, Number(metrics.num_samples) || 0);
        accumulators[clientId].participationRounds += 1;
        accumulators[clientId].accuracySum += Number(metrics.accuracy) || 0;
        accumulators[clientId].f1Sum += Number(metrics.f1_score) || 0;
        accumulators[clientId].throughputSum += Number(metrics.samples_per_second) || 0;
        accumulators[clientId].trainingTimeSum += Number(metrics.training_time) || 0;
      });
    });

    const values = Object.values(accumulators);
    const totalSamples = values.reduce((sum, item) => sum + item.totalSamples, 0);
    const maxAverageThroughput = Math.max(
      0,
      ...values.map(item => item.throughputSum / Math.max(1, item.participationRounds))
    );

    return values.map(item => {
      const rounds = Math.max(1, item.participationRounds);
      const avgAccuracy = item.accuracySum / rounds;
      const avgF1Score = item.f1Sum / rounds;
      const avgThroughput = item.throughputSum / rounds;
      const sampleScore = totalSamples > 0 ? (item.totalSamples / totalSamples) * 100 : 0;
      const participationScore = history.length > 0 ? (item.participationRounds / history.length) * 100 : 0;
      const performanceScore = clampPercent(((avgAccuracy + avgF1Score) / 2) * 100);
      const efficiencyScore = maxAverageThroughput > 0 ? (avgThroughput / maxAverageThroughput) * 100 : 0;
      const contributionScore =
        sampleScore * CONTRIBUTION_WEIGHTS.sample
        + participationScore * CONTRIBUTION_WEIGHTS.participation
        + performanceScore * CONTRIBUTION_WEIGHTS.performance
        + efficiencyScore * CONTRIBUTION_WEIGHTS.efficiency;

      return {
        clientId: item.clientId,
        contributionScore: Number(contributionScore.toFixed(2)),
        sampleContribution: Number(sampleScore.toFixed(2)),
        participationContribution: Number(participationScore.toFixed(2)),
        performanceContribution: Number(performanceScore.toFixed(2)),
        efficiencyContribution: Number(efficiencyScore.toFixed(2)),
        totalSamples: item.totalSamples,
        participationRounds: item.participationRounds,
        avgAccuracy,
        avgF1Score,
        avgThroughput
      };
    }).sort((a, b) => b.contributionScore - a.contributionScore);
  };

  const buildSampleDistributionData = (clientPerformance = []) => (
    [...clientPerformance]
      .sort((a, b) => (b.samples || 0) - (a.samples || 0))
      .map(item => ({
        clientId: item.clientId,
        samples: item.samples || 0
      }))
  );

  const getChartWidth = (rowCount) => Math.max(CLIENT_CHART_MIN_WIDTH, rowCount * CLIENT_BAR_WIDTH);

  const renderScrollableBarChart = (data, height, renderChart) => (
    <div style={{ width: '100%', overflowX: 'auto', overflowY: 'hidden' }}>
      <div style={{ width: getChartWidth(data.length), height }}>
        <ResponsiveContainer width="100%" height="100%">
          {renderChart(data)}
        </ResponsiveContainer>
      </div>
    </div>
  );

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

  const historyTableData = chartData.trainingCurves.map(item => ({
    key: item.round,
    round: item.round,
    accuracy: item.accuracy,
    precision: item.precision,
    recall: item.recall,
    f1Score: item.f1Score,
    balancedAccuracy: item.balancedAccuracy,
    loss: item.loss,
    samplesPerSecond: item.samplesPerSecond
  }));

  const metricColumns = [
    { title: '轮次', dataIndex: 'round', key: 'round' },
    { title: '准确率', dataIndex: 'accuracy', key: 'accuracy', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: 'Precision', dataIndex: 'precision', key: 'precision', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: 'Recall', dataIndex: 'recall', key: 'recall', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: 'F1 Score', dataIndex: 'f1Score', key: 'f1Score', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: 'Balanced Acc', dataIndex: 'balancedAccuracy', key: 'balancedAccuracy', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: '损失', dataIndex: 'loss', key: 'loss', render: value => typeof value === 'number' ? value.toFixed(4) : '-' },
    { title: 'Samples/s', dataIndex: 'samplesPerSecond', key: 'samplesPerSecond', render: value => typeof value === 'number' ? value.toFixed(2) : '-' }
  ];

  const sampleDistributionData = buildSampleDistributionData(chartData.clientPerformance);
  const sampleTableData = [...chartData.clientPerformance]
    .sort((a, b) => getClientSortValue(a.clientId) - getClientSortValue(b.clientId))
    .map(item => ({
      key: item.clientId,
      clientId: item.clientId,
      samples: item.samples || 0,
      accuracy: item.accuracy || 0,
      samplesPerSecond: item.samplesPerSecond || 0
    }));

  const sampleColumns = [
    { title: '客户端', dataIndex: 'clientId', key: 'clientId', sorter: (a, b) => getClientSortValue(a.clientId) - getClientSortValue(b.clientId) },
    { title: '样本数', dataIndex: 'samples', key: 'samples', sorter: (a, b) => a.samples - b.samples },
    { title: '准确率', dataIndex: 'accuracy', key: 'accuracy', render: value => `${((value || 0) * 100).toFixed(2)}%` },
    { title: 'Samples/s', dataIndex: 'samplesPerSecond', key: 'samplesPerSecond', render: value => Number(value || 0).toFixed(2) }
  ];

  const contributionColumns = [
    { title: '排名', key: 'rank', render: (_, record, index) => index + 1 },
    { title: '客户端', dataIndex: 'clientId', key: 'clientId' },
    { title: '综合贡献度', dataIndex: 'contributionScore', key: 'contributionScore', render: value => `${Number(value || 0).toFixed(2)} / 100` },
    { title: '样本贡献', dataIndex: 'sampleContribution', key: 'sampleContribution', render: value => `${Number(value || 0).toFixed(2)}%` },
    { title: '参与贡献', dataIndex: 'participationContribution', key: 'participationContribution', render: value => `${Number(value || 0).toFixed(2)}%` },
    { title: '表现贡献', dataIndex: 'performanceContribution', key: 'performanceContribution', render: value => `${Number(value || 0).toFixed(2)}%` },
    { title: '效率贡献', dataIndex: 'efficiencyContribution', key: 'efficiencyContribution', render: value => `${Number(value || 0).toFixed(2)}%` }
  ];

  const pagedClientDistribution = chartData.clientDistribution.slice(
    (distributionPage - 1) * DISTRIBUTION_PAGE_SIZE,
    distributionPage * DISTRIBUTION_PAGE_SIZE
  );

  return (
    <PageContainer>
      <h1>可视化分析</h1>

      <Space style={{ marginBottom: 16 }}>
        <span>训练记录：</span>
        <Select
          value={selectedRunId || undefined}
          onChange={handleRunChange}
          disabled={trainingRuns.length === 0}
          placeholder="暂无历史训练记录"
          style={{ width: 360 }}
        >
          {trainingRuns.map(run => (
            <Select.Option key={run.id} value={run.id}>
              {`${run.timestamp ? new Date(run.timestamp).toLocaleString() : '历史记录'} | ${(run.dataset_name || '').toUpperCase()} | ${run.aggregation_algorithm || '-'} | ${((run.final_accuracy || 0) * 100).toFixed(2)}%`}
            </Select.Option>
          ))}
        </Select>
        <Button onClick={() => loadVisualizationData(selectedRunId)}>刷新数据</Button>
      </Space>

      {trainingRuns.length === 0 && (
        <ChartCard>
          <Empty description="暂无历史训练记录，请先完成一次训练" />
        </ChartCard>
      )}

      {selectedRunSummary && (
        <ChartCard title="历史训练概览">
          <Row gutter={16}>
            <Col span={4}>
              <Statistic title="最终准确率" value={(selectedRunSummary.finalAccuracy || 0) * 100} precision={2} suffix="%" />
            </Col>
            <Col span={4}>
              <Statistic title="最终损失" value={selectedRunSummary.finalLoss || 0} precision={4} />
            </Col>
            <Col span={4}>
              <Statistic title="F1 Score" value={(selectedRunSummary.finalF1Score || 0) * 100} precision={2} suffix="%" />
            </Col>
            <Col span={4}>
              <Statistic title="训练轮次" value={selectedRunSummary.rounds || 0} />
            </Col>
            <Col span={4}>
              <Statistic title="客户端数" value={selectedRunSummary.numClients || 0} />
            </Col>
            <Col span={4}>
              <Space direction="vertical">
                <span>{(selectedRunSummary.datasetName || '').toUpperCase()} / {selectedRunSummary.modelName || '-'}</span>
                <span>{selectedRunSummary.algorithm || '-'} / {formatIidLabel(selectedRunSummary.iid)}</span>
                <Tag color={selectedRunSummary.status === 'Completed' ? 'green' : selectedRunSummary.status === 'Error' ? 'red' : 'blue'}>{selectedRunSummary.status}</Tag>
              </Space>
            </Col>
          </Row>
        </ChartCard>
      )}

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
          <Row gutter={16}>
            <Col span={12}>
              <ChartCard title="全局分类指标">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData.trainingCurves}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="precision" stroke="#1677ff" strokeWidth={2} name="Precision" dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="recall" stroke="#722ed1" strokeWidth={2} name="Recall" dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="f1Score" stroke="#fa8c16" strokeWidth={2} name="F1 Score" dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="balancedAccuracy" stroke="#13c2c2" strokeWidth={2} name="Balanced Accuracy" dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </Col>
            <Col span={12}>
              <ChartCard title="全局推理吞吐量">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData.trainingCurves}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="samplesPerSecond" stroke="#eb2f96" strokeWidth={2} name="Samples/s" dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </Col>
          </Row>
          <ChartCard title="每轮指标明细">
            <Table
              columns={metricColumns}
              dataSource={historyTableData}
              pagination={{ pageSize: 8 }}
              scroll={{ x: true }}
            />
          </ChartCard>
        </Tabs.TabPane>

        <Tabs.TabPane
          tab={<span><BarChartOutlined />客户端性能</span>}
          key="performance"
        >
          <ChartCard title="客户端贡献度排行">
            {chartData.clientContributions.length > 0 ? (
              renderScrollableBarChart(chartData.clientContributions, 420, data => (
                <BarChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={90} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip cursor={false} formatter={(value) => Number(value || 0).toFixed(2)} />
                  <Legend />
                  <Bar dataKey="contributionScore" fill="#1890ff" name="综合贡献度" />
                </BarChart>
              ))
            ) : (
              <Empty description="暂无贡献度数据" />
            )}
          </ChartCard>

          <ChartCard title="客户端贡献度拆解">
            {chartData.clientContributions.length > 0 ? (
              <>
                {renderScrollableBarChart(chartData.clientContributions, 420, data => (
                  <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={90} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip cursor={false} formatter={(value) => `${Number(value || 0).toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="sampleContribution" fill="#52c41a" name="样本贡献" />
                    <Bar dataKey="participationContribution" fill="#fa8c16" name="参与贡献" />
                    <Bar dataKey="performanceContribution" fill="#722ed1" name="表现贡献" />
                    <Bar dataKey="efficiencyContribution" fill="#13c2c2" name="效率贡献" />
                  </BarChart>
                ))}
                <Table
                  columns={contributionColumns}
                  dataSource={chartData.clientContributions.map(item => ({ ...item, key: item.clientId }))}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: true }}
                />
              </>
            ) : (
              <Empty description="暂无贡献度拆解数据" />
            )}
          </ChartCard>

          <ChartCard title="客户端准确率对比">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData.clientPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={80} />
                <YAxis domain={[0, 1]} />
                <Tooltip cursor={false} />
                <Legend />
                <Bar dataKey="accuracy" fill="#52c41a" name="准确率" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="客户端分类指标对比">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData.clientPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={80} />
                <YAxis domain={[0, 1]} />
                <Tooltip cursor={false} />
                <Legend />
                <Bar dataKey="precision" fill="#1677ff" name="Precision" />
                <Bar dataKey="recall" fill="#722ed1" name="Recall" />
                <Bar dataKey="f1Score" fill="#fa8c16" name="F1 Score" />
                <Bar dataKey="balancedAccuracy" fill="#13c2c2" name="Balanced Accuracy" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="客户端吞吐量对比">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData.clientPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip cursor={false} />
                <Legend />
                <Bar dataKey="samplesPerSecond" fill="#eb2f96" name="Samples/s" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="客户端样本数量分布">
            {sampleDistributionData.length > 0 ? (
              <>
                {renderScrollableBarChart(sampleDistributionData, 420, data => (
                  <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="clientId" angle={-45} textAnchor="end" height={90} />
                    <YAxis />
                    <Tooltip cursor={false} />
                    <Legend />
                    <Bar dataKey="samples" fill="#1677ff" name="样本数" />
                  </BarChart>
                ))}
                <Table
                  columns={sampleColumns}
                  dataSource={sampleTableData}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: true }}
                />
              </>
            ) : (
              <Empty description="暂无客户端样本数量数据" />
            )}
          </ChartCard>
        </Tabs.TabPane>

        <Tabs.TabPane
          tab={<span><PieChartOutlined />数据分布</span>}
          key="distribution"
        >
          <ChartCard title="客户端类别分布">
            {chartData.clientDistribution.length > 0 ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                  <span style={{ color: '#6e6e73' }}>
                    共 {chartData.clientDistribution.length} 个客户端，每页 {DISTRIBUTION_PAGE_SIZE} 个
                  </span>
                  <Pagination
                    current={distributionPage}
                    pageSize={DISTRIBUTION_PAGE_SIZE}
                    total={chartData.clientDistribution.length}
                    showSizeChanger={false}
                    onChange={setDistributionPage}
                  />
                </div>
                {pagedClientDistribution.map(client => (
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
                <Pagination
                  current={distributionPage}
                  pageSize={DISTRIBUTION_PAGE_SIZE}
                  total={chartData.clientDistribution.length}
                  showSizeChanger={false}
                  onChange={setDistributionPage}
                  style={{ textAlign: 'right' }}
                />
              </>
            ) : (
              <Empty description="暂无客户端类别分布数据" />
            )}
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
