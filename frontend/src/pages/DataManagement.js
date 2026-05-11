import React, { useState, useEffect } from 'react';
import { Card, Button, Select, Table, Tag, message, Divider, Space, Input, Row, Col, Statistic, Form, InputNumber, Switch, Upload, Popconfirm } from 'antd';
import { UploadOutlined, DatabaseOutlined, PlayCircleOutlined, DeleteOutlined, InfoCircleOutlined, SettingOutlined, TeamOutlined, CloudUploadOutlined, BarChartOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const PageContainer = styled.div`
  padding: 0;
`;

const DatasetCard = styled(Card)`
  margin-bottom: 20px;
  border-radius: 28px;
`;

const DataManagement = () => {
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('mnist');
  const [federatedConfig, setFederatedConfig] = useState({
    num_clients: 10,
    batch_size: 64,
    iid: true,
    client_fraction: 0.5,
    non_iid_alpha: 0.5,
    non_iid_seed: 42
  });
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [allDatasetInfos, setAllDatasetInfos] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadAvailableDatasets();
    loadUploadedDatasets();
  }, []);

  useEffect(() => {
    loadDatasetInfo();
  }, [selectedDataset]);

  const loadAvailableDatasets = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/data/datasets');
      const data = await response.json();
      const datasets = data.datasets || [];
      setAvailableDatasets(datasets);
      
      // 加载所有数据集的详细信息用于表格展示
      const infos = await Promise.all(datasets.map(async (name) => {
        const resp = await fetch(`http://localhost:5000/api/data/datasets/${name}/info`);
        return await resp.json();
      }));
      setAllDatasetInfos(infos);
    } catch (error) {
      message.error('无法获取数据集列表');
    }
  };

  const loadDatasetInfo = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/data/datasets/${selectedDataset}/info`);
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      message.error('无法获取数据集信息');
    }
  };

  const loadUploadedDatasets = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/data/uploads');
      const data = await response.json();
      if (response.ok) {
        setUploadedFiles(data.files || []);
      }
    } catch (error) {
      message.error('无法获取上传数据集列表');
    }
  };

  const handleDatasetUpload = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/api/data/uploads', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (response.ok) {
        message.success(`${file.name} 上传成功`);
        loadAvailableDatasets();
        loadUploadedDatasets();
        onSuccess(data, file);
      } else {
        message.error(`上传失败: ${data.error}`);
        onError(new Error(data.error));
      }
    } catch (error) {
      message.error('上传失败，请检查后端连接');
      onError(error);
    }
  };

  const deleteUploadedDataset = async (filename) => {
    try {
      const response = await fetch(`http://localhost:5000/api/data/uploads/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      const data = await response.json();
      if (response.ok) {
        message.success('上传数据集文件已删除');
        loadAvailableDatasets();
        loadUploadedDatasets();
      } else {
        message.error(`删除失败: ${data.error}`);
      }
    } catch (error) {
      message.error('删除上传数据集文件失败');
    }
  };

  const handleDatasetLoad = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/data/datasets/${selectedDataset}/load`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          batch_size: federatedConfig.batch_size,
          train: true
        })
      });
      const data = await response.json();
      if (response.ok) {
        message.success(`数据集 ${selectedDataset} 加载成功`);
      } else {
        message.error(`加载数据集失败: ${data.error || '未知错误'}`);
      }
    } catch (error) {
      message.error('加载数据集失败');
    } finally {
      setLoading(false);
    }
  };

  const handleFederatedSetup = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/data/federated/setup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          dataset_name: selectedDataset,
          ...federatedConfig
        })
      });
      const data = await response.json();
      if (response.ok) {
        message.success(`联邦学习环境设置成功 - ${data.num_clients}个客户端`);
      } else {
        message.error(`联邦学习环境设置失败: ${data.error || '未知错误'}`);
      }
    } catch (error) {
      message.error('联邦学习环境设置失败');
    } finally {
      setLoading(false);
    }
  };

  const datasetColumns = [
    {
      title: '数据集名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '样本数量',
      dataIndex: 'num_samples',
      key: 'num_samples',
    },
    {
      title: '类别数量',
      dataIndex: 'num_classes',
      key: 'num_classes',
    },
    {
      title: '状态',
      key: 'status',
      render: () => <Tag color="green">已准备就绪</Tag>
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => startTrainingWithDataset(record.dataset_id)}
          >
            开始训练
          </Button>
          <Button
            icon={<InfoCircleOutlined />}
            onClick={() => showDatasetDetails(record)}
          >
            详情
          </Button>
        </Space>
      )
    }
  ];

  const uploadedColumns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename'
    },
    {
      title: '大小',
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      render: value => `${((value || 0) / 1024).toFixed(2)} KB`
    },
    {
      title: '上传时间',
      dataIndex: 'modified_time',
      key: 'modified_time',
      render: value => value ? new Date(value).toLocaleString() : '-'
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record) => record.custom_dataset?.trainable
        ? <Tag color="green">可训练</Tag>
        : <Tag color="orange">{record.custom_dataset?.parse_error || '仅存储'}</Tag>
    },
    {
      title: '数据集ID',
      key: 'dataset_id',
      render: (_, record) => record.custom_dataset?.trainable ? record.custom_dataset.id : '-'
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          {record.custom_dataset?.trainable && (
            <Button
              type="primary"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => startTrainingWithDataset(record.custom_dataset.id)}
            >
              训练
            </Button>
          )}
          <Popconfirm
            title="确认删除这个上传文件？"
            okText="删除"
            cancelText="取消"
            onConfirm={() => deleteUploadedDataset(record.filename)}
          >
            <Button danger icon={<DeleteOutlined />} size="small">删除</Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const startTrainingWithDataset = (datasetName) => {
    navigate('/train', { state: { selectedDataset: datasetName } });
  };

  const showDatasetDetails = (dataset) => {
    setSelectedDataset(dataset.dataset_id);
    setDatasetInfo(dataset);
  };

  const datasetData = allDatasetInfos.map(info => ({
    key: info.id || (info.name || '').toLowerCase(),
    dataset_id: info.id || (info.name || '').toLowerCase(),
    name: info.name,
    num_samples: info.num_samples,
    num_classes: info.num_classes,
    input_shape: info.input_shape,
    classes: info.classes
  }));

  return (
    <PageContainer>
      <Row gutter={24}>
        <Col span={24}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <h1 style={{ margin: 0 }}>数据管理</h1>
            <Space>
              <Upload
                accept=".csv,.json,.jsonl,.npz,.npy,.zip,.tar,.gz,.pt,.pth,.pkl"
                showUploadList={false}
                customRequest={handleDatasetUpload}
              >
                <Button icon={<CloudUploadOutlined />}>导入数据集</Button>
              </Upload>
              <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => navigate('/train')}>
                跳转至训练
              </Button>
            </Space>
          </div>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={10}>
          <DatasetCard title="选择数据集">
            <Form layout="vertical">
              <Form.Item label="可用数据集">
                <Select
                  value={selectedDataset}
                  onChange={(value) => {
                    setSelectedDataset(value);
                  }}
                  style={{ width: '100%' }}
                >
                  {availableDatasets.map(dataset => (
                    <Select.Option key={dataset} value={dataset}>
                      {dataset.toUpperCase()}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
              <Button
                type="primary"
                icon={<DatabaseOutlined />}
                onClick={handleDatasetLoad}
                loading={loading}
                block
              >
                加载并检查数据集
              </Button>
            </Form>

            {datasetInfo && (
              <div style={{ marginTop: 24 }}>
                <Divider orientation="left">数据集详情</Divider>
                <Row gutter={[0, 12]}>
                  <Col span={12}><Statistic title="样本总数" value={datasetInfo.num_samples} /></Col>
                  <Col span={12}><Statistic title="类别数量" value={datasetInfo.num_classes} /></Col>
                  <Col span={24}>
                    <div style={{ color: 'rgba(0, 0, 0, 0.45)', marginBottom: 4 }}>输入形状</div>
                    <Tag color="blue">{datasetInfo.input_shape?.join(' × ')}</Tag>
                  </Col>
                  <Col span={24}>
                    <div style={{ color: 'rgba(0, 0, 0, 0.45)', marginBottom: 4 }}>类别列表</div>
                    <Space wrap>
                      {datasetInfo.classes?.map(cls => (
                        <Tag key={cls}>{cls}</Tag>
                      ))}
                    </Space>
                  </Col>
                </Row>
              </div>
            )}
          </DatasetCard>
        </Col>

        <Col span={14}>
          <DatasetCard title="联邦学习环境配置">
            <Form
              layout="vertical"
              initialValues={federatedConfig}
              onValuesChange={(changed) => setFederatedConfig({...federatedConfig, ...changed})}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="客户端数量" name="num_clients">
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="批次大小" name="batch_size">
                    <InputNumber min={16} max={256} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="客户端参与比例" name="client_fraction">
                    <InputNumber min={0.1} max={1} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="数据分布" name="iid" valuePropName="checked">
                    <Switch checkedChildren="IID (独立同分布)" unCheckedChildren="Non-IID (非独立同分布)" />
                  </Form.Item>
                </Col>
              </Row>
              {!federatedConfig.iid && (
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Dirichlet α" name="non_iid_alpha">
                      <InputNumber min={0.01} max={100} step={0.1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Non-IID随机种子" name="non_iid_seed">
                      <InputNumber min={0} max={999999} step={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              )}
              <Divider />
              <Button
                type="primary"
                size="large"
                icon={<SettingOutlined />}
                onClick={handleFederatedSetup}
                loading={loading}
                block
              >
                应用联邦学习配置
              </Button>
            </Form>
          </DatasetCard>

          <DatasetCard title="可用数据集列表">
            <Table
              columns={datasetColumns}
              dataSource={datasetData}
              pagination={false}
              rowKey="name"
              size="small"
            />
          </DatasetCard>

          <DatasetCard title="上传数据集文件">
            <Upload
              accept=".csv,.json,.jsonl,.npz,.npy,.zip,.tar,.gz,.pt,.pth,.pkl"
              showUploadList={false}
              customRequest={handleDatasetUpload}
            >
              <Button icon={<UploadOutlined />} style={{ marginBottom: 16 }}>选择文件上传</Button>
            </Upload>
            <Table
              columns={uploadedColumns}
              dataSource={uploadedFiles}
              pagination={{ pageSize: 5 }}
              rowKey="filename"
              size="small"
            />
          </DatasetCard>
        </Col>
      </Row>
    </PageContainer>
  );
};

export default DataManagement;