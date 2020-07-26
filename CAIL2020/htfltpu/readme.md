#Introduction
从网上获取的合同模板，根据网上标注的分类，将合同分为24类。一部分数据没有分类，作为测试数据集。
使用清洗过的308.5MB，59266行标注数据作为训练集；将2400行数据作为验证集，验证集参与训练，24类每类100条。

训练过程如图1所示： 

![ALT](science/htfl-train.svg)


#BERT ACC:79.73%
#CNN  ACC:32.67%

#https://colab.research.google.com/drive/1SZ62rhHlP5x3w_XwXndbvicPiCxT44IM#scrollTo=HaQK5Xa2FhbK


Metric: CompileTime
  TotalSamples: 6
  Accumulator: 337ms683.150us
  ValueRate: 002ms340.836us / second
  Rate: 0.0417158 / second
  Percentiles: 1%=021ms262.351us; 5%=021ms262.351us; 10%=021ms262.351us; 20%=028ms297.071us; 50%=034ms308.611us; 80%=109ms552.633us; 90%=115ms911.973us; 95%=115ms911.973us; 99%=115ms911.973us
Metric: DeviceLockWait
  TotalSamples: 6015
  Accumulator: 01m21s477ms157.009us
  ValueRate: 364ms964.691us / second
  Rate: 31.0544 / second
  Percentiles: 1%=001.210us; 5%=001.291us; 10%=001.369us; 20%=001.480us; 50%=591.810us; 80%=025ms678.922us; 90%=025ms106.944us; 95%=025ms432.093us; 99%=026ms817.643us
Metric: ExecuteTime
  TotalSamples: 6012
  Accumulator: 04m06s157ms341.565us
  ValueRate: 909ms005.660us / second
  Rate: 31.0566 / second
  Percentiles: 1%=019ms324.275us; 5%=020ms610.525us; 10%=020ms814.245us; 20%=020ms108.413us; 50%=037ms225.870us; 80%=038ms264.060us; 90%=039ms547.779us; 95%=039ms839.730us; 99%=040ms122.669us
Metric: InboundData
  TotalSamples: 2412
  Accumulator: 18.80KB
  ValueRate: 124.39B / second
  Rate: 15.5493 / second
  Percentiles: 1%=8.00B; 5%=8.00B; 10%=8.00B; 20%=8.00B; 50%=8.00B; 80%=8.00B; 90%=8.00B; 95%=8.00B; 99%=8.00B
Metric: InputOutputAliasCount
  TotalSamples: 6
  Accumulator: 2126.00
  ValueRate: 14.78 / second
  Rate: 0.0417186 / second
  Percentiles: 1%=1.00; 5%=1.00; 10%=1.00; 20%=1.00; 50%=311.00; 80%=806.00; 90%=806.00; 95%=806.00; 99%=806.00
Metric: IrValueTensorToXlaData
  TotalSamples: 201
  Accumulator: 04s309ms381.931us
  ValueRate: 01s161ms275.340us / second
  Rate: 54.1647 / second
  Percentiles: 1%=001ms045.810us; 5%=001ms166.480us; 10%=001ms234.870us; 20%=001ms426.260us; 50%=002ms817.440us; 80%=024ms244.831us; 90%=091ms239.791us; 95%=092ms888.341us; 99%=095ms697.152us
Metric: OutboundData
  TotalSamples: 2316
  Accumulator: 474.61MB
  ValueRate: 337.60KB / second
  Rate: 5.26875 / second
  Percentiles: 1%=8.00B; 5%=8.00B; 10%=8.00B; 20%=8.00B; 50%=96.06KB; 80%=96.06KB; 90%=96.06KB; 95%=96.06KB; 99%=96.06KB
Metric: ReleaseDataHandlesTime
  TotalSamples: 15142
  Accumulator: 01m22s124ms749.690us
  ValueRate: 427ms644.154us / second
  Rate: 65.0037 / second
  Percentiles: 1%=745.150us; 5%=891.819us; 10%=964.000us; 20%=001ms093.939us; 50%=002ms342.809us; 80%=020ms353.086us; 90%=024ms588.114us; 95%=024ms042.533us; 99%=024ms474.644us
Metric: TensorsGraphSize
  TotalSamples: 6013
  Accumulator: 22527705.00
  ValueRate: 58165.10 / second
  Rate: 31.0545 / second
  Percentiles: 1%=1871.00; 5%=1871.00; 10%=1871.00; 20%=1871.00; 50%=1875.00; 80%=1875.00; 90%=1875.00; 95%=1875.00; 99%=1875.00
Metric: TransferFromServerTime
  TotalSamples: 2412
  Accumulator: 05s845ms940.793us
  ValueRate: 031ms870.681us / second
  Rate: 15.5493 / second
  Percentiles: 1%=001ms419.011us; 5%=002ms533.969us; 10%=002ms595.859us; 20%=002ms700.479us; 50%=002ms946.480us; 80%=002ms286.568us; 90%=002ms424.219us; 95%=003ms519.859us; 99%=003ms691.409us
Metric: TransferToServerTime
  TotalSamples: 2316
  Accumulator: 02m37s765ms193.337us
  ValueRate: 213ms335.462us / second
  Rate: 5.27013 / second
  Percentiles: 1%=003ms708.170us; 5%=003ms951.419us; 10%=003ms150.320us; 20%=004ms183.960us; 50%=037ms174.070us; 80%=081ms687.131us; 90%=084ms031.771us; 95%=087ms272.423us; 99%=088ms196.840us
Metric: TransferToServerTransformTime
  TotalSamples: 2316
  Accumulator: 02s393ms018.002us
  ValueRate: 009ms912.598us / second
  Rate: 5.26875 / second
  Percentiles: 1%=094.990us; 5%=111.290us; 10%=125.849us; 20%=133.381us; 50%=002ms709.399us; 80%=003ms900.600us; 90%=003ms454.669us; 95%=004ms594.900us; 99%=006ms901.980us
Counter: CachedCompile
  Value: 6007
Counter: CreateCompileHandles
  Value: 6
Counter: CreateDataHandles
  Value: 1731718
Counter: CreateXlaTensor
  Value: 5846004
Counter: DestroyDataHandles
  Value: 1730895
Counter: DestroyXlaTensor
  Value: 5845198
Counter: DeviceDataCacheMiss
  Value: 15
Counter: MarkStep
  Value: 3603
Counter: ReleaseDataHandles
  Value: 1730895
Counter: UncachedCompile
  Value: 6
Counter: XRTAllocateFromTensor_Empty
  Value: 45
Counter: XrtCompile_Empty
  Value: 144
Counter: XrtExecuteChained_Empty
  Value: 144
Counter: XrtExecute_Empty
  Value: 144
Counter: XrtRead_Empty
  Value: 144
Counter: XrtReleaseAllocationHandle_Empty
  Value: 144
Counter: XrtReleaseCompileHandle_Empty
  Value: 144
Counter: XrtSessionCount
  Value: 11
Counter: XrtSubTuple_Empty
  Value: 144
Counter: aten::_local_scalar_dense
  Value: 2412
Counter: xla::_log_softmax
  Value: 1200
Counter: xla::_log_softmax_backward_data
  Value: 1200
Counter: xla::_softmax
  Value: 46800
Counter: xla::_softmax_backward_data
  Value: 15600
Counter: xla::_unsafe_view
  Value: 345600
Counter: xla::add
  Value: 194400
Counter: xla::add_
  Value: 1223799
Counter: xla::addcdiv_
  Value: 241200
Counter: xla::addcmul
  Value: 90000
Counter: xla::addcmul_
  Value: 241200
Counter: xla::addmm
  Value: 7200
Counter: xla::arange_out
  Value: 3600
Counter: xla::as_strided
  Value: 201
Counter: xla::bernoulli_
  Value: 45600
Counter: xla::bmm
  Value: 144000
Counter: xla::copy_
  Value: 6201
Counter: xla::div
  Value: 57600
Counter: xla::div_
  Value: 45600
Counter: xla::embedding
  Value: 10800
Counter: xla::embedding_dense_backward
  Value: 3600
Counter: xla::empty
  Value: 57003
Counter: xla::empty_strided
  Value: 201
Counter: xla::eq
  Value: 2400
Counter: xla::expand
  Value: 176400
Counter: xla::fill_
  Value: 1200
Counter: xla::gelu
  Value: 43200
Counter: xla::gelu_backward
  Value: 14400
Counter: xla::index_select
  Value: 10800
Counter: xla::max
  Value: 2400
Counter: xla::mm
  Value: 436800
Counter: xla::mul
  Value: 184800
Counter: xla::mul_
  Value: 482400
Counter: xla::native_batch_norm
  Value: 90000
Counter: xla::native_batch_norm_backward
  Value: 30000
Counter: xla::native_layer_norm
  Value: 90000
Counter: xla::native_layer_norm_backward
  Value: 30000
Counter: xla::nll_loss_backward
  Value: 1200
Counter: xla::nll_loss_forward
  Value: 1200
Counter: xla::permute
  Value: 230400
Counter: xla::rsub
  Value: 3600
Counter: xla::select
  Value: 4800
Counter: xla::slice
  Value: 12000
Counter: xla::sqrt
  Value: 241200
Counter: xla::sub
  Value: 30000
Counter: xla::sum
  Value: 151200
Counter: xla::t
  Value: 532800
Counter: xla::tanh
  Value: 3600
Counter: xla::tanh_backward
  Value: 1200
Counter: xla::transpose
  Value: 115200
Counter: xla::unsqueeze
  Value: 10800
Counter: xla::view
  Value: 1804800
Counter: xla::zero_
  Value: 243801
Metric: XrtAllocateFromTensor
  TotalSamples: 54498
  Accumulator: 04m03s422ms966.466us
  Mean: 004ms341.600us
  StdDev: 798.823us
  Rate: 62.7036 / second
  Percentiles: 25%=004ms801.503us; 50%=004ms306.117us; 80%=005ms954.763us; 90%=005ms393.150us; 95%=006ms899.343us; 99%=006ms403.459us
Metric: XrtCompile
  TotalSamples: 30
  Accumulator: 57s053ms885.151us
  Mean: 02s902ms762.838us
  StdDev: 05s236ms004.381us
  Rate: 0.00338187 / second
  Percentiles: 25%=013ms013.938us; 50%=053ms804.884us; 80%=237ms551.015us; 90%=08s274ms499.091us; 95%=20s138ms772.218us; 99%=21s814ms917.470us
Metric: XrtExecute
  TotalSamples: 16452
  Accumulator: 14m08s175ms163.004us
  Mean: 028ms701.718us
  StdDev: 009ms913.179us
  Rate: 31.0565 / second
  Percentiles: 25%=019ms764.127us; 50%=036ms661.613us; 80%=037ms583.539us; 90%=037ms825.437us; 95%=037ms045.499us; 99%=038ms283.086us
Metric: XrtExecutorEvict
  TotalSamples: 0
  Accumulator: nanB
  Mean: nanB
  StdDev: nanB
  Percentiles: 
Metric: XrtReadLiteral
  TotalSamples: 4870
  Accumulator: 04s294ms699.556us
  Mean: 884.411us
  StdDev: 254.653us
  Rate: 15.5493 / second
  Percentiles: 25%=685.259us; 50%=791.495us; 80%=001ms175.546us; 90%=001ms287.954us; 95%=001ms341.758us; 99%=001ms426.014us
Metric: XrtReleaseAllocation
  TotalSamples: 50257
  Accumulator: 50s885ms267.790us
  Mean: 640.177us
  StdDev: 735.305us
  Rate: 65.0524 / second
  Percentiles: 25%=031.367us; 50%=105.846us; 80%=001ms457.374us; 90%=002ms789.876us; 95%=002ms008.494us; 99%=002ms275.096us


#1 core TPU 7 minutes
#8 core TPU 5 minutes
2020-07-25 15:29:43.136294: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-07-25 15:29:53.204238: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.228285: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.257802: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.292715: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.324211: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.360339: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
2020-07-25 15:29:53.394303: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) d8ff2e405352:56851
Loading train records for train...
2020-07-25 15:30:01.585279: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
Loading train records for train...
2020-07-25 15:30:02.081152: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
Loading train records for train...
2020-07-25 15:30:02.312634: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
2020-07-25 15:30:02.391050: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
Loading train records for train...
2020-07-25 15:30:02.472371: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
Loading train records for train...
Loading train records for train...
2020-07-25 15:30:02.927869: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
2020-07-25 15:30:03.059762: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at d8ff2e405352:56851
Loading train records for train...
Loading train records for train...
2400it [00:09, 250.09it/s]
217it [00:00, 237.33it/s]2400 training records loaded.
Loading train records for valid...
2400it [00:09, 243.66it/s]
2320it [00:09, 220.11it/s]2400 training records loaded.
Loading train records for valid...
2400it [00:09, 243.71it/s]
2400it [00:09, 255.10it/s]
2400it [00:09, 258.14it/s]
2389it [00:09, 258.24it/s]2400 training records loaded.
Loading train records for valid...
2400it [00:09, 255.99it/s]
2400it [00:09, 241.18it/s]
2280it [00:09, 244.52it/s]2400 training records loaded.
Loading train records for valid...
2353it [00:09, 226.90it/s]2400 training records loaded.
Loading train records for valid...
2400it [00:09, 242.69it/s]
2400 training records loaded.
Loading train records for valid...
2400 training records loaded.
Loading train records for valid...
2400 training records loaded.
Loading train records for valid...
2400it [00:09, 245.98it/s]
2400 train records loaded.
Loading valid records...
2400it [00:09, 251.82it/s]
2219it [00:08, 264.48it/s]2400 train records loaded.
1916it [00:07, 244.02it/s]Loading valid records...
2400it [00:09, 254.57it/s]
2400it [00:09, 251.83it/s]
2400it [00:09, 250.34it/s]
2230it [00:08, 257.05it/s]2400 train records loaded.
Loading valid records...
2026it [00:08, 265.71it/s]2400 train records loaded.
Loading valid records...
2400it [00:09, 249.72it/s]
2143it [00:08, 268.06it/s]2400 train records loaded.
Loading valid records...
2400it [00:09, 253.31it/s]
2363it [00:09, 248.34it/s]2400 train records loaded.
Loading valid records...
2400it [00:09, 251.44it/s]
2400 train records loaded.
Loading valid records...
2400 train records loaded.
Loading valid records...
2400it [00:09, 252.67it/s]
2400 valid records loaded.
2400it [00:09, 260.99it/s]
2400it [00:09, 258.46it/s]
2187it [00:09, 230.34it/s]2400 valid records loaded.
2400it [00:09, 247.18it/s]
1579it [00:06, 243.00it/s]2400 valid records loaded.
2400it [00:10, 235.64it/s]
1994it [00:07, 235.28it/s]2400 valid records loaded.
1809it [00:07, 272.91it/s]2400 valid records loaded.
2400it [00:09, 256.50it/s]
2400it [00:08, 296.18it/s]
2400 valid records loaded.
2300it [00:08, 535.03it/s]2400 valid records loaded.
2400it [00:08, 284.89it/s]
2400 valid records loaded.
[xla:0](0) Loss=2.83388 Rate=0.08 GlobalRate=0.08 Time=Sat Jul 25 15:39:04 2020
[xla:1](0) Loss=2.85225 Rate=0.59 GlobalRate=0.59 Time=Sat Jul 25 15:39:43 2020
[xla:4](0) Loss=2.82860 Rate=0.80 GlobalRate=0.80 Time=Sat Jul 25 15:39:46 2020
[xla:5](0) Loss=2.83566 Rate=1.09 GlobalRate=1.09 Time=Sat Jul 25 15:39:46 2020
[xla:2](0) Loss=2.84216 Rate=1.15 GlobalRate=1.15 Time=Sat Jul 25 15:39:47 2020
[xla:3](0) Loss=2.83738 Rate=1.52 GlobalRate=1.52 Time=Sat Jul 25 15:39:48 2020
[xla:6](0) Loss=2.82494 Rate=1.20 GlobalRate=1.20 Time=Sat Jul 25 15:39:48 2020
[xla:7](0) Loss=2.81731 Rate=2.31 GlobalRate=2.31 Time=Sat Jul 25 15:39:49 2020
[xla:5](100) Loss=2.92953 Rate=0.86 GlobalRate=0.72 Time=Sat Jul 25 15:44:27 2020
[xla:4](100) Loss=2.92953 Rate=0.75 GlobalRate=0.71 Time=Sat Jul 25 15:44:27 2020
[xla:1](100) Loss=2.42953 Rate=0.66 GlobalRate=0.70 Time=Sat Jul 25 15:44:27 2020
[xla:7](100) Loss=2.92953 Rate=1.36 GlobalRate=0.72 Time=Sat Jul 25 15:44:27 2020
[xla:2](100) Loss=2.42953 Rate=0.89 GlobalRate=0.72 Time=Sat Jul 25 15:44:27 2020
[xla:0](100) Loss=2.92953 Rate=0.40 GlobalRate=0.58 Time=Sat Jul 25 15:44:27 2020
[xla:3](100) Loss=2.92953 Rate=1.04 GlobalRate=0.72 Time=Sat Jul 25 15:44:27 2020
[xla:6](100) Loss=2.92953 Rate=0.91 GlobalRate=0.72 Time=Sat Jul 25 15:44:27 2020
Finished training epoch 1
[xla:2] Accuracy=20.83%
[xla:3] Accuracy=20.83%
[xla:1] Accuracy=20.83%
[xla:6] Accuracy=20.83%
[xla:7] Accuracy=20.83%
[xla:4] Accuracy=20.83%
[xla:0] Accuracy=20.83%
Finished test epoch 1 20.833333333333332
[xla:5] Accuracy=20.83%