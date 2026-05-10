import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Select, Input, Table, Tag, message, Divider, Space, Progress, Tabs, Row, Col, Statistic, Form, InputNumber, Switch, Popconfirm } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined, SaveOutlined, LineChartOutlined, DashboardOutlined, SettingOutlined, DeleteOutlined } from '@ant-design/icons';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const PageContainer = styled.div`
  padding: 20px;
`;

const TrainingCard = styled(Card)`
  margin-bottom: 16px;
`;

const StatusIndicator = styled.div`
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 8px;
  background-color: ${props => props.status === 'running' ? '#52c41a' : props.status === 'stopped' ? '#ff4d4f' : '#fa8c16'};
`;

const AlgorithmDescription = styled.div`
  margin-bottom: 24px;
  padding: 12px;
  border: 1px solid #f0f0f0;
  border-radius: 6px;
  background: #fafafa;
`;

const AlgorithmTitle = styled.div`
  margin-bottom: 6px;
  font-weight: 600;
  color: #262626;
`;

const AlgorithmText = styled.div`
  color: #595959;
  line-height: 1.6;
`;

const AlgorithmMeta = styled.div`
  margin-top: 8px;
  color: #8c8c8c;
  font-size: 12px;
  line-height: 1.5;
`;

const AGGREGATION_ALGORITHM_DETAILS = {
  fedavg: {
    label: 'FedAvg',
    shortDescription: '按客户端样本数加权平均模型参数，是联邦学习中最基础、最常用的聚合方法。',
    scenario: '适合客户端数据分布相对稳定、训练环境较均衡的基线实验。',
    parameters: '主要关注客户端比例、训练轮数和本地 batch size。'
  },
  fedprox: {
    label: 'FedProx',
    shortDescription: '在客户端本地训练中加入近端约束，减少本地模型偏离全局模型过远。',
    scenario: '适合 Non-IID 数据、客户端算力或数据质量差异较大的场景。',
    parameters: '可通过 FedProx μ 调整约束强度，μ 越大，本地更新越保守。'
  },
  fedavgm: {
    label: 'FedAvgM',
    shortDescription: '在服务端聚合更新时加入动量项，让全局模型更新更平滑。',
    scenario: '适合训练曲线震荡明显，或希望在 FedAvg 基础上提升收敛稳定性的场景。',
    parameters: '可通过 FedAvgM 动量控制历史更新的影响，常用值接近 0.9。'
  },
  fedadam: {
    label: 'FedAdam',
    shortDescription: '在服务端使用 Adam 风格的自适应优化，根据一阶和二阶矩调整聚合步长。',
    scenario: '适合梯度尺度变化较大、希望提升收敛速度的训练任务。',
    parameters: '关注服务端学习率、β1、β2 和 τ，学习率过大可能带来震荡。'
  },
  fedyogi: {
    label: 'FedYogi',
    shortDescription: 'FedAdam 的变体，对二阶矩更新更保守，避免自适应项持续增大。',
    scenario: '适合客户端更新差异明显、FedAdam 不够稳定的场景。',
    parameters: '关注 β1、β2、τ 与服务端学习率，通常比 FedAdam 更稳健。'
  },
  fedadagrad: {
    label: 'FedAdagrad',
    shortDescription: '在服务端累积历史平方更新，为频繁变化的参数自动降低学习率。',
    scenario: '适合稀疏更新或早期需要快速适应、后期希望逐渐稳定的任务。',
    parameters: '主要调节服务端学习率和 τ，累计项会让后续更新趋于保守。'
  }
};

const enrichAggregationAlgorithm = (algorithm) => {
  const details = AGGREGATION_ALGORITHM_DETAILS[algorithm.value] || {};

  return {
    ...details,
    ...algorithm,
    label: algorithm.label || details.label || algorithm.value
  };
};

const formatPercent = (value) => typeof value === 'number' ? (value * 100).toFixed(2) + '%' : '-';
const formatNumber = (value, digits = 4) => typeof value === 'number' ? value.toFixed(digits) : '-';
const ADAPTIVE_AGGREGATION_ALGORITHMS = new Set(['fedadam', 'fedyogi', 'fedadagrad']);
const DEFAULT_SERVER_LR = 1.0;
const DEFAULT_ADAPTIVE_SERVER_LR = 0.001;
const MAX_ADAPTIVE_SERVER_LR = 0.01;

const ModelTraining = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [datasets, setDatasets] = useState(['mnist', 'cifar10', 'cifar100']);
  const [selectedDataset, setSelectedDataset] = useState('mnist');
  const [models, setModels] = useState([{ value: 'cnn', label: 'CNN' }]);
  const [aggregationAlgorithms, setAggregationAlgorithms] = useState([
    { value: 'fedavg', label: 'FedAvg' },
    { value: 'fedprox', label: 'FedProx' },
    { value: 'fedavgm', label: 'FedAvgM' },
    { value: 'fedadam', label: 'FedAdam' },
    { value: 'fedyogi', label: 'FedYogi' },
    { value: 'fedadagrad', label: 'FedAdagrad' }
  ].map(enrichAggregationAlgorithm));

  useEffect(() => {
    if (location.state && location.state.selectedDataset) {
      setSelectedDataset(location.state.selectedDataset);
    }
  }, [location.state]);
  const [trainingConfig, setTrainingConfig] = useState({
    num_clients: 10,
    num_rounds: 10,
    client_fraction: 0.5,
    aggregation_algorithm: 'fedavg',
    iid: true,
    batch_size: 64,
    device: 'cuda',
    server_lr: DEFAULT_SERVER_LR,
    server_momentum: 0.9,
    proximal_mu: 0.01,
    adaptive_beta1: 0.9,
    adaptive_beta2: 0.99,
    adaptive_tau: 0.001,
    non_iid_classes_per_client: 2,
    non_iid_seed: 42,
    model_name: 'cnn'
  });
  const [trainingStatus, setTrainingStatus] = useState('stopped'); // stopped, running, paused
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [metrics, setMetrics] = useState({});
  const [activeTab, setActiveTab] = useState('config');
  const [performanceMetrics, setPerformanceMetrics] = useState([]);
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [savedModels, setSavedModels] = useState([]);
  const intervalRef = useRef(null);

  const startStatusPolling = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5000/api/train/status');
        const data = await response.json();

        if (['Running', 'Completed', 'Stopped'].includes(data.status)) {
          setTrainingStatus(data.status === 'Running' ? 'running' : 'stopped');
          setCurrentRound(data.current_round);
          setTrainingHistory(data.history || []); 
          if (data.latest_metrics) {
            setMetrics(data.latest_metrics);
          }
          if (data.status === 'Completed') {
            loadHistory();
          }
          if (data.status !== 'Running') {
            clearInterval(intervalRef.current);
          }
        } else if (data.status === 'Not started') {
          setTrainingStatus('stopped');
          setCurrentRound(0);
          setTrainingHistory([]);
          setMetrics({});
          clearInterval(intervalRef.current);
        } else if (data.status === 'Error') {
          setTrainingStatus('stopped');
          message.error(`训练发生错误: ${data.error}`);
          clearInterval(intervalRef.current);
        }
      } catch (error) {
        console.error('获取训练状态失败', error);
      }
    }, 2000);
  };

  useEffect(() => {
    // 组件挂载时，检查后台是否正在训练
    const checkInitialStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/train/status');
        const data = await response.json();
        if (data.status === 'Running') {
          setTrainingStatus('running');
          startStatusPolling();
        } else if (['Completed', 'Stopped'].includes(data.status) && data.history && data.history.length > 0) {
          setTrainingHistory(data.history);
          setCurrentRound(data.current_round);
          if (data.latest_metrics) {
            setMetrics(data.latest_metrics);
          }
        }
      } catch (error) {
        console.error('检查初始训练状态失败', error);
      }
    };

    const loadAggregationAlgorithms = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/train/algorithms');
        const data = await response.json();
        if (response.ok && data.algorithms) {
          setAggregationAlgorithms(data.algorithms.map(enrichAggregationAlgorithm));
        }
      } catch (error) {
        console.error('获取聚合算法列表失败', error);
      }
    };

    const loadDatasets = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/data/datasets');
        const data = await response.json();
        if (response.ok && data.datasets) {
          setDatasets(data.datasets);
        }
      } catch (error) {
        console.error('获取数据集列表失败', error);
      }
    };

    checkInitialStatus();
    loadAggregationAlgorithms();
    loadDatasets();
    loadHistory();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const loadHistory = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/main/dashboard_stats');
      const data = await response.json();
      if (response.ok) {
        setTrainingRuns(data.training_runs || []);
        setSavedModels(data.saved_models || []);
      }
    } catch (error) {
      console.error('获取历史记录失败', error);
    }
  };

  useEffect(() => {
    // 当训练历史更新时，自动更新性能指标表格
    if (trainingHistory.length > 0) {
      updatePerformanceMetricsTable(trainingHistory);
    }
  }, [trainingHistory]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/model/models?dataset_name=${selectedDataset}`);
        const data = await response.json();
        if (response.ok && data.models) {
          setModels(data.models);
          setTrainingConfig(prevConfig => {
            const hasSelectedModel = data.models.some(model => model.value === prevConfig.model_name);
            return {
              ...prevConfig,
              model_name: hasSelectedModel ? prevConfig.model_name : data.models[0]?.value || 'cnn'
            };
          });
        }
      } catch (error) {
        console.error('获取模型列表失败', error);
      }
    };

    loadModels();
  }, [selectedDataset]);

  const updatePerformanceMetricsTable = (history) => {
    const formattedMetrics = history.map((h, index) => ({
      key: index,
      round: h.round,
      accuracy: formatPercent(h.global_metrics?.accuracy),
      precision: formatPercent(h.global_metrics?.precision),
      recall: formatPercent(h.global_metrics?.recall),
      f1_score: formatPercent(h.global_metrics?.f1_score),
      balanced_accuracy: formatPercent(h.global_metrics?.balanced_accuracy),
      loss: formatNumber(h.global_metrics?.loss),
      samples_per_second: formatNumber(h.global_metrics?.samples_per_second, 2),
      num_samples: h.global_metrics?.num_samples || '-'
    }));
    setPerformanceMetrics(formattedMetrics);
  };

  const startTraining = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/train/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          dataset_name: selectedDataset,
          ...trainingConfig
        })
      });

      const data = await response.json();

      if (response.ok) {
        message.success('训练启动成功！');
        setTrainingStatus('running');
        setCurrentRound(0);
        setTrainingHistory([]); 
        setPerformanceMetrics([]);

        // 开始轮询训练状态
        startStatusPolling();
      } else {
        message.error(`训练启动失败: ${data.error}`);
      }
    } catch (error) {
      message.error('训练启动失败，请检查后端连接');
    }
  };

  const stopTraining = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/train/stop', {
        method: 'POST'
      });
      const data = await response.json();
      if (response.ok) {
        setTrainingStatus('stopped');
        message.info('训练已停止');
      } else {
        message.error(`停止训练失败: ${data.error}`);
      }
    } catch (error) {
      message.error('停止训练失败');
    }
  };

  const saveModel = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/train/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          path: `./checkpoints/${selectedDataset}_model.pth`
        })
      });

      if (response.ok) {
        message.success('模型保存成功！');
        loadHistory();
      } else {
        message.error('模型保存失败');
      }
    } catch (error) {
      message.error('模型保存失败');
    }
  };

  const getMetrics = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/train/status');
      const data = await response.json();
      if (data.history) {
        setTrainingHistory(data.history);
        message.success('指标已更新');
      }
    } catch (error) {
      console.error('获取指标失败', error);
    }
  };

  const deleteTrainingRun = async (runId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/train/history/${encodeURIComponent(runId)}`, {
        method: 'DELETE'
      });
      const data = await response.json();
      if (response.ok) {
        message.success('历史训练记录已删除');
        loadHistory();
      } else {
        message.error(`删除失败: ${data.error}`);
      }
    } catch (error) {
      message.error('删除历史训练记录失败');
    }
  };

  const chartData = trainingHistory.map(h => ({
    round: h.round,
    accuracy: h.global_metrics?.accuracy || 0,
    precision: h.global_metrics?.precision || 0,
    recall: h.global_metrics?.recall || 0,
    f1_score: h.global_metrics?.f1_score || 0,
    balanced_accuracy: h.global_metrics?.balanced_accuracy || 0,
    loss: h.global_metrics?.loss || 0,
    samples_per_second: h.global_metrics?.samples_per_second || 0
  }));

  const selectedAggregationAlgorithm = aggregationAlgorithms.find(
    algorithm => algorithm.value === trainingConfig.aggregation_algorithm
  ) || enrichAggregationAlgorithm({ value: trainingConfig.aggregation_algorithm });

  const configColumns = [
    { title: '参数', dataIndex: 'key', key: 'key', width: '40%' },
    { title: '值', dataIndex: 'value', key: 'value', width: '60%' }
  ];

  const configData = Object.entries({
    ...trainingConfig,
    dataset: selectedDataset.toUpperCase()
  }).map(([key, value]) => ({
    key: key.replace(/_/g, ' ').toUpperCase(),
    value: typeof value === 'boolean' ? (value ? '是' : '否') : value
  }));

  return (
    <PageContainer>
      <h1>模型训练</h1>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <Tabs.TabPane tab={<span><SettingOutlined />训练配置</span>} key="config">
          <Row gutter={16}>
            <Col span={16}>
              <TrainingCard title="训练参数配置">
                <Form
                  form={form}
                  layout="vertical"
                  initialValues={trainingConfig}
                  onValuesChange={(changedValues) => {
                    const nextValues = { ...changedValues };
                    if (Object.prototype.hasOwnProperty.call(changedValues, 'aggregation_algorithm')) {
                      const isAdaptive = ADAPTIVE_AGGREGATION_ALGORITHMS.has(changedValues.aggregation_algorithm);
                      if (isAdaptive && trainingConfig.server_lr > MAX_ADAPTIVE_SERVER_LR) {
                        nextValues.server_lr = DEFAULT_ADAPTIVE_SERVER_LR;
                        form.setFieldsValue({ server_lr: DEFAULT_ADAPTIVE_SERVER_LR });
                      } else if (!isAdaptive && trainingConfig.server_lr === DEFAULT_ADAPTIVE_SERVER_LR) {
                        nextValues.server_lr = DEFAULT_SERVER_LR;
                        form.setFieldsValue({ server_lr: DEFAULT_SERVER_LR });
                      }
                    }
                    setTrainingConfig(prevConfig => ({ ...prevConfig, ...nextValues }));
                  }}
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="数据集" name="dataset_name">
                        <Select value={selectedDataset} onChange={setSelectedDataset}>
                          {datasets.map(dataset => (
                            <Select.Option key={dataset} value={dataset}>{dataset.toUpperCase()}</Select.Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="客户端数量" name="num_clients">
                        <InputNumber min={2} max={100} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="模型架构" name="model_name">
                        <Select
                          value={trainingConfig.model_name}
                          onChange={(value) => setTrainingConfig(prevConfig => ({ ...prevConfig, model_name: value }))}
                        >
                          {models.map(model => (
                            <Select.Option key={model.value} value={model.value}>
                              {model.label}
                            </Select.Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="聚合算法" name="aggregation_algorithm">
                        <Select>
                          {aggregationAlgorithms.map(algorithm => (
                            <Select.Option key={algorithm.value} value={algorithm.value}>
                              <div>
                                <div>{algorithm.label}</div>
                                {algorithm.shortDescription && (
                                  <div style={{ color: '#8c8c8c', fontSize: 12, whiteSpace: 'normal' }}>
                                    {algorithm.shortDescription}
                                  </div>
                                )}
                              </div>
                            </Select.Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="服务端学习率" name="server_lr">
                        <InputNumber
                          min={0.0001}
                          max={ADAPTIVE_AGGREGATION_ALGORITHMS.has(trainingConfig.aggregation_algorithm) ? MAX_ADAPTIVE_SERVER_LR : 10}
                          step={ADAPTIVE_AGGREGATION_ALGORITHMS.has(trainingConfig.aggregation_algorithm) ? 0.001 : 0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={24}>
                      <AlgorithmDescription>
                        <AlgorithmTitle>{selectedAggregationAlgorithm.label}</AlgorithmTitle>
                        <AlgorithmText>{selectedAggregationAlgorithm.shortDescription}</AlgorithmText>
                        {selectedAggregationAlgorithm.scenario && (
                          <AlgorithmMeta>适用场景：{selectedAggregationAlgorithm.scenario}</AlgorithmMeta>
                        )}
                        {selectedAggregationAlgorithm.parameters && (
                          <AlgorithmMeta>参数提示：{selectedAggregationAlgorithm.parameters}</AlgorithmMeta>
                        )}
                      </AlgorithmDescription>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="训练轮次" name="num_rounds">
                        <InputNumber min={1} max={100} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="客户端参与比例" name="client_fraction">
                        <InputNumber min={0.1} max={1} step={0.1} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="批次大小" name="batch_size">
                        <InputNumber min={16} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="数据分布" name="iid" valuePropName="checked">
                        <Switch checkedChildren="IID" unCheckedChildren="Non-IID" />
                      </Form.Item>
                    </Col>
                  </Row>
                  {!trainingConfig.iid && (
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="每客户端类别数" name="non_iid_classes_per_client">
                          <InputNumber min={1} max={100} step={1} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="Non-IID随机种子" name="non_iid_seed">
                          <InputNumber min={0} max={999999} step={1} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                    </Row>
                  )}
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="FedAvgM 动量" name="server_momentum">
                        <InputNumber min={0} max={0.999} step={0.05} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="FedProx μ" name="proximal_mu">
                        <InputNumber min={0} max={1} step={0.001} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="自适应 β1" name="adaptive_beta1">
                        <InputNumber min={0} max={0.999} step={0.05} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="自适应 β2" name="adaptive_beta2">
                        <InputNumber min={0} max={0.999} step={0.01} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="自适应 τ" name="adaptive_tau">
                        <InputNumber min={0.000001} max={1} step={0.001} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>
              </TrainingCard>
            </Col>
            <Col span={8}>
              <TrainingCard title="训练控制">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button type="primary" icon={<PlayCircleOutlined />} onClick={startTraining} disabled={trainingStatus === 'running'} size="large" block>开始训练</Button>
                  <Button icon={<PauseCircleOutlined />} onClick={() => setTrainingStatus('paused')} disabled={trainingStatus !== 'running'} size="large" block>暂停训练</Button>
                  <Button danger icon={<StopOutlined />} onClick={stopTraining} disabled={trainingStatus === 'stopped'} size="large" block>停止训练</Button>
                  <Button icon={<SaveOutlined />} onClick={saveModel} disabled={trainingStatus !== 'stopped' || currentRound === 0} size="large" block>保存模型</Button>
                </Space>
                <Divider />
                <div style={{ marginBottom: 8 }}>
                  <StatusIndicator status={trainingStatus} />
                  状态：<Tag color={trainingStatus === 'running' ? 'green' : trainingStatus === 'stopped' ? 'red' : 'orange'}>
                    {trainingStatus === 'running' ? '训练中' : trainingStatus === 'stopped' ? '已停止' : '已暂停'}
                  </Tag>
                </div>
                <div>当前轮次：<strong>{currentRound}</strong> / {trainingConfig.num_rounds}</div>
                {typeof metrics.accuracy === 'number' && (
                  <div style={{ marginTop: 8 }}>
                    <Statistic title="准确率" value={metrics.accuracy * 100} precision={2} suffix="%" valueStyle={{ color: '#3f8600' }} />
                    <Statistic title="损失" value={metrics.loss} precision={4} valueStyle={{ color: '#cf1322' }} />
                    <Statistic title="F1 Score" value={(metrics.f1_score || 0) * 100} precision={2} suffix="%" valueStyle={{ color: '#1677ff' }} />
                  </div>
                )}
              </TrainingCard>
            </Col>
          </Row>
          <TrainingCard title="配置概览">
            <Table columns={configColumns} dataSource={configData} pagination={false} size="small" />
          </TrainingCard>
        </Tabs.TabPane>

        <Tabs.TabPane tab={<span><LineChartOutlined />训练监控</span>} key="monitoring">
          <Row gutter={16}>
            <Col span={12}>
              <TrainingCard title="准确率曲线">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#52c41a" strokeWidth={2} name="准确率" />
                  </LineChart>
                </ResponsiveContainer>
              </TrainingCard>
            </Col>
            <Col span={12}>
              <TrainingCard title="损失曲线">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#ff4d4f" strokeWidth={2} name="损失" />
                  </LineChart>
                </ResponsiveContainer>
              </TrainingCard>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <TrainingCard title="分类指标曲线">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="precision" stroke="#1677ff" strokeWidth={2} name="Precision" />
                    <Line type="monotone" dataKey="recall" stroke="#722ed1" strokeWidth={2} name="Recall" />
                    <Line type="monotone" dataKey="f1_score" stroke="#fa8c16" strokeWidth={2} name="F1 Score" />
                    <Line type="monotone" dataKey="balanced_accuracy" stroke="#13c2c2" strokeWidth={2} name="Balanced Accuracy" />
                  </LineChart>
                </ResponsiveContainer>
              </TrainingCard>
            </Col>
            <Col span={12}>
              <TrainingCard title="吞吐量曲线">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="samples_per_second" stroke="#eb2f96" strokeWidth={2} name="Samples/s" />
                  </LineChart>
                </ResponsiveContainer>
              </TrainingCard>
            </Col>
          </Row>

          <TrainingCard title="性能指标" style={{ marginTop: 16 }}>
            <Button onClick={getMetrics} style={{ marginBottom: 16 }}>刷新指标</Button>
            <Table
              columns={[
                { title: '轮次', dataIndex: 'round', key: 'round' },
                { title: '准确率', dataIndex: 'accuracy', key: 'accuracy' },
                { title: 'Precision', dataIndex: 'precision', key: 'precision' },
                { title: 'Recall', dataIndex: 'recall', key: 'recall' },
                { title: 'F1 Score', dataIndex: 'f1_score', key: 'f1_score' },
                { title: 'Balanced Acc', dataIndex: 'balanced_accuracy', key: 'balanced_accuracy' },
                { title: '损失', dataIndex: 'loss', key: 'loss' },
                { title: 'Samples/s', dataIndex: 'samples_per_second', key: 'samples_per_second' },
                { title: '样本数', dataIndex: 'num_samples', key: 'num_samples' }
              ]}
              dataSource={performanceMetrics}
              pagination={false}
              scroll={{ x: true }}
            />
          </TrainingCard>
        </Tabs.TabPane>

        <Tabs.TabPane tab={<span><SaveOutlined />历史记录</span>} key="history">
          <Row gutter={16}>
            <Col span={14}>
              <TrainingCard title="历史训练记录">
                <Button onClick={loadHistory} style={{ marginBottom: 16 }}>刷新历史</Button>
                <Table
                  columns={[
                    { title: '时间', dataIndex: 'timestamp', key: 'timestamp', render: value => value ? new Date(value).toLocaleString() : '-' },
                    { title: '数据集', dataIndex: 'dataset_name', key: 'dataset_name', render: value => (value || '').toUpperCase() },
                    { title: '模型', dataIndex: 'model_name', key: 'model_name' },
                    { title: '算法', dataIndex: 'aggregation_algorithm', key: 'aggregation_algorithm' },
                    { title: '轮次', dataIndex: 'rounds', key: 'rounds' },
                    { title: '准确率', dataIndex: 'final_accuracy', key: 'final_accuracy', render: value => `${((value || 0) * 100).toFixed(2)}%` },
                    { title: '状态', dataIndex: 'status', key: 'status', render: value => <Tag color={value === 'Completed' ? 'green' : value === 'Error' ? 'red' : 'blue'}>{value}</Tag> },
                    {
                      title: '操作',
                      key: 'action',
                      render: (_, record) => (
                        <Space>
                          <Button
                            size="small"
                            icon={<LineChartOutlined />}
                            onClick={() => navigate('/visualization', { state: { selectedRunId: record.id } })}
                          >
                            可视化
                          </Button>
                          <Popconfirm
                            title="确认删除这条历史训练记录？"
                            okText="删除"
                            cancelText="取消"
                            onConfirm={() => deleteTrainingRun(record.id)}
                          >
                            <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
                          </Popconfirm>
                        </Space>
                      )
                    }
                  ]}
                  dataSource={trainingRuns}
                  rowKey="id"
                  pagination={{ pageSize: 5 }}
                  scroll={{ x: true }}
                />
              </TrainingCard>
            </Col>
            <Col span={10}>
              <TrainingCard title="历史模型">
                <Table
                  columns={[
                    { title: '文件名', dataIndex: 'filename', key: 'filename' },
                    { title: '数据集', dataIndex: 'dataset_name', key: 'dataset_name', render: value => (value || 'unknown').toUpperCase() },
                    { title: '模型', key: 'model', render: (_, record) => record.model_name || record.model_class || 'model' },
                    { title: '轮次', dataIndex: 'rounds', key: 'rounds' }
                  ]}
                  dataSource={savedModels}
                  rowKey="id"
                  pagination={{ pageSize: 5 }}
                  scroll={{ x: true }}
                />
              </TrainingCard>
            </Col>
          </Row>
        </Tabs.TabPane>
      </Tabs>
    </PageContainer>
  );
};

export default ModelTraining;
