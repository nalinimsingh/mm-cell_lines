??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
v
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d-*
shared_namedense1/kernel
o
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes

:d-*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:-*
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

:--*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:-*
dtype0
x
encoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*
shared_nameencoder/kernel
q
"encoder/kernel/Read/ReadVariableOpReadVariableOpencoder/kernel*
_output_shapes

:-*
dtype0
p
encoder/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencoder/bias
i
 encoder/bias/Read/ReadVariableOpReadVariableOpencoder/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
z
dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d-* 
shared_namedense1/kernel/m
s
#dense1/kernel/m/Read/ReadVariableOpReadVariableOpdense1/kernel/m*
_output_shapes

:d-*
dtype0
r
dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense1/bias/m
k
!dense1/bias/m/Read/ReadVariableOpReadVariableOpdense1/bias/m*
_output_shapes
:-*
dtype0
z
dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense2/kernel/m
s
#dense2/kernel/m/Read/ReadVariableOpReadVariableOpdense2/kernel/m*
_output_shapes

:--*
dtype0
r
dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense2/bias/m
k
!dense2/bias/m/Read/ReadVariableOpReadVariableOpdense2/bias/m*
_output_shapes
:-*
dtype0
|
encoder/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*!
shared_nameencoder/kernel/m
u
$encoder/kernel/m/Read/ReadVariableOpReadVariableOpencoder/kernel/m*
_output_shapes

:-*
dtype0
t
encoder/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencoder/bias/m
m
"encoder/bias/m/Read/ReadVariableOpReadVariableOpencoder/bias/m*
_output_shapes
:*
dtype0
x
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel/m
q
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes

:*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:*
dtype0
z
dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d-* 
shared_namedense1/kernel/v
s
#dense1/kernel/v/Read/ReadVariableOpReadVariableOpdense1/kernel/v*
_output_shapes

:d-*
dtype0
r
dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense1/bias/v
k
!dense1/bias/v/Read/ReadVariableOpReadVariableOpdense1/bias/v*
_output_shapes
:-*
dtype0
z
dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense2/kernel/v
s
#dense2/kernel/v/Read/ReadVariableOpReadVariableOpdense2/kernel/v*
_output_shapes

:--*
dtype0
r
dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense2/bias/v
k
!dense2/bias/v/Read/ReadVariableOpReadVariableOpdense2/bias/v*
_output_shapes
:-*
dtype0
|
encoder/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*!
shared_nameencoder/kernel/v
u
$encoder/kernel/v/Read/ReadVariableOpReadVariableOpencoder/kernel/v*
_output_shapes

:-*
dtype0
t
encoder/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencoder/bias/v
m
"encoder/bias/v/Read/ReadVariableOpReadVariableOpencoder/bias/v*
_output_shapes
:*
dtype0
x
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel/v
q
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes

:*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
?mPmQmRmS!mT"mU'mV(mWvXvYvZv[!v\"v]'v^(v_
8
0
1
2
3
!4
"5
'6
(7
8
0
1
2
3
!4
"5
'6
(7
 
?
-layer_metrics
.layer_regularization_losses
/non_trainable_variables
	variables

0layers
	trainable_variables
1metrics

regularization_losses
 
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
2layer_metrics
3non_trainable_variables

4layers
	variables
regularization_losses
trainable_variables
5metrics
6layer_regularization_losses
 
 
 
?
7layer_metrics
8non_trainable_variables

9layers
	variables
regularization_losses
trainable_variables
:metrics
;layer_regularization_losses
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
<layer_metrics
=non_trainable_variables

>layers
	variables
regularization_losses
trainable_variables
?metrics
@layer_regularization_losses
 
 
 
?
Alayer_metrics
Bnon_trainable_variables

Clayers
	variables
regularization_losses
trainable_variables
Dmetrics
Elayer_regularization_losses
ZX
VARIABLE_VALUEencoder/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEencoder/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?
Flayer_metrics
Gnon_trainable_variables

Hlayers
#	variables
$regularization_losses
%trainable_variables
Imetrics
Jlayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?
Klayer_metrics
Lnon_trainable_variables

Mlayers
)	variables
*regularization_losses
+trainable_variables
Nmetrics
Olayer_regularization_losses
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEdense1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEdense1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEdense2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEdense2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEencoder/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEencoder/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEdense1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEdense1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEdense2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEdense2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEencoder/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEencoder/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense1_inputPlaceholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense1_inputdense1/kerneldense1/biasdense2/kerneldense2/biasencoder/kernelencoder/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_41355
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp"encoder/kernel/Read/ReadVariableOp encoder/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp#dense1/kernel/m/Read/ReadVariableOp!dense1/bias/m/Read/ReadVariableOp#dense2/kernel/m/Read/ReadVariableOp!dense2/bias/m/Read/ReadVariableOp$encoder/kernel/m/Read/ReadVariableOp"encoder/bias/m/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp#dense1/kernel/v/Read/ReadVariableOp!dense1/bias/v/Read/ReadVariableOp#dense2/kernel/v/Read/ReadVariableOp!dense2/bias/v/Read/ReadVariableOp$encoder/kernel/v/Read/ReadVariableOp"encoder/bias/v/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_41702
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasdense2/kerneldense2/biasencoder/kernelencoder/biasdense/kernel
dense/biasdense1/kernel/mdense1/bias/mdense2/kernel/mdense2/bias/mencoder/kernel/mencoder/bias/mdense/kernel/mdense/bias/mdense1/kernel/vdense1/bias/vdense2/kernel/vdense2/bias/vencoder/kernel/vencoder/bias/vdense/kernel/vdense/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_41784˂
?
{
&__inference_dense2_layer_call_fn_41542

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_411122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_41194

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense1_layer_call_and_return_conditional_losses_41486

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d-*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????-2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

b
C__inference_dropout1_layer_call_and_return_conditional_losses_41507

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_41237
dense1_input
dense1_41214
dense1_41216
dense2_41220
dense2_41222
encoder_41226
encoder_41228
dense_41231
dense_41233
identity??dense/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCalldense1_inputdense1_41214dense1_41216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_410552 
dense1/StatefulPartitionedCall?
dropout1/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410882
dropout1/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0dense2_41220dense2_41222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_411122 
dense2/StatefulPartitionedCall?
dropout2/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411452
dropout2/PartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0encoder_41226encoder_41228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_411682!
encoder/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0dense_41231dense_41233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_411942
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input
?
D
(__inference_dropout1_layer_call_fn_41522

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_41475

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_413132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_41332
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_413132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input
?
a
C__inference_dropout1_layer_call_and_return_conditional_losses_41088

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????-2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????-2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?g
?
!__inference__traced_restore_41784
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias$
 assignvariableop_2_dense2_kernel"
assignvariableop_3_dense2_bias%
!assignvariableop_4_encoder_kernel#
assignvariableop_5_encoder_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias&
"assignvariableop_8_dense1_kernel_m$
 assignvariableop_9_dense1_bias_m'
#assignvariableop_10_dense2_kernel_m%
!assignvariableop_11_dense2_bias_m(
$assignvariableop_12_encoder_kernel_m&
"assignvariableop_13_encoder_bias_m&
"assignvariableop_14_dense_kernel_m$
 assignvariableop_15_dense_bias_m'
#assignvariableop_16_dense1_kernel_v%
!assignvariableop_17_dense1_bias_v'
#assignvariableop_18_dense2_kernel_v%
!assignvariableop_19_dense2_bias_v(
$assignvariableop_20_encoder_kernel_v&
"assignvariableop_21_encoder_bias_v&
"assignvariableop_22_dense_kernel_v$
 assignvariableop_23_dense_bias_v
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_encoder_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_encoder_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense1_kernel_mIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense1_bias_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense2_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense2_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_encoder_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_encoder_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense1_kernel_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense1_bias_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense2_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense2_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_encoder_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_encoder_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
A__inference_dense1_layer_call_and_return_conditional_losses_41055

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d-*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????-2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?%
?
E__inference_sequential_layer_call_and_return_conditional_losses_41433

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource*
&encoder_matmul_readvariableop_resource+
'encoder_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?encoder/BiasAdd/ReadVariableOp?encoder/MatMul/ReadVariableOp?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:d-*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
dense1/Sigmoidx
dropout1/IdentityIdentitydense1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????-2
dropout1/Identity?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:--*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldropout1/Identity:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense2/BiasAddv
dense2/SigmoidSigmoiddense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
dense2/Sigmoidx
dropout2/IdentityIdentitydense2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????-2
dropout2/Identity?
encoder/MatMul/ReadVariableOpReadVariableOp&encoder_matmul_readvariableop_resource*
_output_shapes

:-*
dtype02
encoder/MatMul/ReadVariableOp?
encoder/MatMulMatMuldropout2/Identity:output:0%encoder/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/MatMul?
encoder/BiasAdd/ReadVariableOpReadVariableOp'encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
encoder/BiasAdd/ReadVariableOp?
encoder/BiasAddBiasAddencoder/MatMul:product:0&encoder/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/BiasAdd?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulencoder/BiasAdd:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^encoder/BiasAdd/ReadVariableOp^encoder/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2@
encoder/BiasAdd/ReadVariableOpencoder/BiasAdd/ReadVariableOp2>
encoder/MatMul/ReadVariableOpencoder/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

b
C__inference_dropout1_layer_call_and_return_conditional_losses_41083

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
a
C__inference_dropout2_layer_call_and_return_conditional_losses_41559

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????-2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????-2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
D
(__inference_dropout2_layer_call_fn_41569

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?

b
C__inference_dropout2_layer_call_and_return_conditional_losses_41554

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?9
?	
__inference__traced_save_41702
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop-
)savev2_encoder_kernel_read_readvariableop+
'savev2_encoder_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop.
*savev2_dense1_kernel_m_read_readvariableop,
(savev2_dense1_bias_m_read_readvariableop.
*savev2_dense2_kernel_m_read_readvariableop,
(savev2_dense2_bias_m_read_readvariableop/
+savev2_encoder_kernel_m_read_readvariableop-
)savev2_encoder_bias_m_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop.
*savev2_dense1_kernel_v_read_readvariableop,
(savev2_dense1_bias_v_read_readvariableop.
*savev2_dense2_kernel_v_read_readvariableop,
(savev2_dense2_bias_v_read_readvariableop/
+savev2_encoder_kernel_v_read_readvariableop-
)savev2_encoder_bias_v_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop)savev2_encoder_kernel_read_readvariableop'savev2_encoder_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop*savev2_dense1_kernel_m_read_readvariableop(savev2_dense1_bias_m_read_readvariableop*savev2_dense2_kernel_m_read_readvariableop(savev2_dense2_bias_m_read_readvariableop+savev2_encoder_kernel_m_read_readvariableop)savev2_encoder_bias_m_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop*savev2_dense1_kernel_v_read_readvariableop(savev2_dense1_bias_v_read_readvariableop*savev2_dense2_kernel_v_read_readvariableop(savev2_dense2_bias_v_read_readvariableop+savev2_encoder_kernel_v_read_readvariableop)savev2_encoder_bias_v_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d-:-:--:-:-::::d-:-:--:-:-::::d-:-:--:-:-:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:-: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:d-: 


_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:-: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:d-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:-: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
#__inference_signature_wrapper_41355
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_410402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_41313

inputs
dense1_41290
dense1_41292
dense2_41296
dense2_41298
encoder_41302
encoder_41304
dense_41307
dense_41309
identity??dense/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_41290dense1_41292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_410552 
dense1/StatefulPartitionedCall?
dropout1/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410882
dropout1/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0dense2_41296dense2_41298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_411122 
dense2/StatefulPartitionedCall?
dropout2/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411452
dropout2/PartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0encoder_41302encoder_41304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_411682!
encoder/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0dense_41307dense_41309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_411942
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
{
&__inference_dense1_layer_call_fn_41495

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_410552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?.
?
 __inference__wrapped_model_41040
dense1_input4
0sequential_dense1_matmul_readvariableop_resource5
1sequential_dense1_biasadd_readvariableop_resource4
0sequential_dense2_matmul_readvariableop_resource5
1sequential_dense2_biasadd_readvariableop_resource5
1sequential_encoder_matmul_readvariableop_resource6
2sequential_encoder_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?(sequential/dense1/BiasAdd/ReadVariableOp?'sequential/dense1/MatMul/ReadVariableOp?(sequential/dense2/BiasAdd/ReadVariableOp?'sequential/dense2/MatMul/ReadVariableOp?)sequential/encoder/BiasAdd/ReadVariableOp?(sequential/encoder/MatMul/ReadVariableOp?
'sequential/dense1/MatMul/ReadVariableOpReadVariableOp0sequential_dense1_matmul_readvariableop_resource*
_output_shapes

:d-*
dtype02)
'sequential/dense1/MatMul/ReadVariableOp?
sequential/dense1/MatMulMatMuldense1_input/sequential/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
sequential/dense1/MatMul?
(sequential/dense1/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense1_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02*
(sequential/dense1/BiasAdd/ReadVariableOp?
sequential/dense1/BiasAddBiasAdd"sequential/dense1/MatMul:product:00sequential/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
sequential/dense1/BiasAdd?
sequential/dense1/SigmoidSigmoid"sequential/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
sequential/dense1/Sigmoid?
sequential/dropout1/IdentityIdentitysequential/dense1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????-2
sequential/dropout1/Identity?
'sequential/dense2/MatMul/ReadVariableOpReadVariableOp0sequential_dense2_matmul_readvariableop_resource*
_output_shapes

:--*
dtype02)
'sequential/dense2/MatMul/ReadVariableOp?
sequential/dense2/MatMulMatMul%sequential/dropout1/Identity:output:0/sequential/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
sequential/dense2/MatMul?
(sequential/dense2/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense2_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02*
(sequential/dense2/BiasAdd/ReadVariableOp?
sequential/dense2/BiasAddBiasAdd"sequential/dense2/MatMul:product:00sequential/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
sequential/dense2/BiasAdd?
sequential/dense2/SigmoidSigmoid"sequential/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
sequential/dense2/Sigmoid?
sequential/dropout2/IdentityIdentitysequential/dense2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????-2
sequential/dropout2/Identity?
(sequential/encoder/MatMul/ReadVariableOpReadVariableOp1sequential_encoder_matmul_readvariableop_resource*
_output_shapes

:-*
dtype02*
(sequential/encoder/MatMul/ReadVariableOp?
sequential/encoder/MatMulMatMul%sequential/dropout2/Identity:output:00sequential/encoder/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/encoder/MatMul?
)sequential/encoder/BiasAdd/ReadVariableOpReadVariableOp2sequential_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/encoder/BiasAdd/ReadVariableOp?
sequential/encoder/BiasAddBiasAdd#sequential/encoder/MatMul:product:01sequential/encoder/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/encoder/BiasAdd?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/encoder/BiasAdd:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
IdentityIdentity!sequential/dense/BiasAdd:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense1/BiasAdd/ReadVariableOp(^sequential/dense1/MatMul/ReadVariableOp)^sequential/dense2/BiasAdd/ReadVariableOp(^sequential/dense2/MatMul/ReadVariableOp*^sequential/encoder/BiasAdd/ReadVariableOp)^sequential/encoder/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense1/BiasAdd/ReadVariableOp(sequential/dense1/BiasAdd/ReadVariableOp2R
'sequential/dense1/MatMul/ReadVariableOp'sequential/dense1/MatMul/ReadVariableOp2T
(sequential/dense2/BiasAdd/ReadVariableOp(sequential/dense2/BiasAdd/ReadVariableOp2R
'sequential/dense2/MatMul/ReadVariableOp'sequential/dense2/MatMul/ReadVariableOp2V
)sequential/encoder/BiasAdd/ReadVariableOp)sequential/encoder/BiasAdd/ReadVariableOp2T
(sequential/encoder/MatMul/ReadVariableOp(sequential/encoder/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input
?
|
'__inference_encoder_layer_call_fn_41588

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_411682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_41598

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
E__inference_sequential_layer_call_and_return_conditional_losses_41401

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource*
&encoder_matmul_readvariableop_resource+
'encoder_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?encoder/BiasAdd/ReadVariableOp?encoder/MatMul/ReadVariableOp?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:d-*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
dense1/Sigmoidu
dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout1/dropout/Const?
dropout1/dropout/MulMuldense1/Sigmoid:y:0dropout1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout1/dropout/Mulr
dropout1/dropout/ShapeShapedense1/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout1/dropout/Shape?
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02/
-dropout1/dropout/random_uniform/RandomUniform?
dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2!
dropout1/dropout/GreaterEqual/y?
dropout1/dropout/GreaterEqualGreaterEqual6dropout1/dropout/random_uniform/RandomUniform:output:0(dropout1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout1/dropout/GreaterEqual?
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout1/dropout/Cast?
dropout1/dropout/Mul_1Muldropout1/dropout/Mul:z:0dropout1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout1/dropout/Mul_1?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:--*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldropout1/dropout/Mul_1:z:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
dense2/BiasAddv
dense2/SigmoidSigmoiddense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????-2
dense2/Sigmoidu
dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout2/dropout/Const?
dropout2/dropout/MulMuldense2/Sigmoid:y:0dropout2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout2/dropout/Mulr
dropout2/dropout/ShapeShapedense2/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout2/dropout/Shape?
-dropout2/dropout/random_uniform/RandomUniformRandomUniformdropout2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02/
-dropout2/dropout/random_uniform/RandomUniform?
dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2!
dropout2/dropout/GreaterEqual/y?
dropout2/dropout/GreaterEqualGreaterEqual6dropout2/dropout/random_uniform/RandomUniform:output:0(dropout2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout2/dropout/GreaterEqual?
dropout2/dropout/CastCast!dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout2/dropout/Cast?
dropout2/dropout/Mul_1Muldropout2/dropout/Mul:z:0dropout2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout2/dropout/Mul_1?
encoder/MatMul/ReadVariableOpReadVariableOp&encoder_matmul_readvariableop_resource*
_output_shapes

:-*
dtype02
encoder/MatMul/ReadVariableOp?
encoder/MatMulMatMuldropout2/dropout/Mul_1:z:0%encoder/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/MatMul?
encoder/BiasAdd/ReadVariableOpReadVariableOp'encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
encoder/BiasAdd/ReadVariableOp?
encoder/BiasAddBiasAddencoder/MatMul:product:0&encoder/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/BiasAdd?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulencoder/BiasAdd:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^encoder/BiasAdd/ReadVariableOp^encoder/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2@
encoder/BiasAdd/ReadVariableOpencoder/BiasAdd/ReadVariableOp2>
encoder/MatMul/ReadVariableOpencoder/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
(__inference_dropout1_layer_call_fn_41517

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_41266

inputs
dense1_41243
dense1_41245
dense2_41249
dense2_41251
encoder_41255
encoder_41257
dense_41260
dense_41262
identity??dense/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall? dropout1/StatefulPartitionedCall? dropout2/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_41243dense1_41245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_410552 
dense1/StatefulPartitionedCall?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410832"
 dropout1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0dense2_41249dense2_41251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_411122 
dense2/StatefulPartitionedCall?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411402"
 dropout2/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0encoder_41255encoder_41257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_411682!
encoder/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0dense_41260dense_41262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_411942
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_41211
dense1_input
dense1_41066
dense1_41068
dense2_41123
dense2_41125
encoder_41179
encoder_41181
dense_41205
dense_41207
identity??dense/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall? dropout1/StatefulPartitionedCall? dropout2/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCalldense1_inputdense1_41066dense1_41068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_410552 
dense1/StatefulPartitionedCall?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout1_layer_call_and_return_conditional_losses_410832"
 dropout1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0dense2_41123dense2_41125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_411122 
dense2/StatefulPartitionedCall?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411402"
 dropout2/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0encoder_41179encoder_41181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_411682!
encoder/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0dense_41205dense_41207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_411942
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input
?
z
%__inference_dense_layer_call_fn_41607

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_411942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
(__inference_dropout2_layer_call_fn_41564

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout2_layer_call_and_return_conditional_losses_411402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?

b
C__inference_dropout2_layer_call_and_return_conditional_losses_41140

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????-2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????-*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????-2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????-2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????-2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
A__inference_dense2_layer_call_and_return_conditional_losses_41533

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????-2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
B__inference_encoder_layer_call_and_return_conditional_losses_41168

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
a
C__inference_dropout2_layer_call_and_return_conditional_losses_41145

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????-2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????-2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_41454

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_412662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
C__inference_dropout1_layer_call_and_return_conditional_losses_41512

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????-2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????-2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????-:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
B__inference_encoder_layer_call_and_return_conditional_losses_41579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
A__inference_dense2_layer_call_and_return_conditional_losses_41112

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????-2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????-2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_41285
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_412662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????d
&
_user_specified_namedense1_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
dense1_input5
serving_default_dense1_input:0?????????d9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?-
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"?*
_tf_keras_sequential?*{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense1_input"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "encoder", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense1_input"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "encoder", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 45, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 45}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45]}}
?
	variables
regularization_losses
trainable_variables
 	keras_api
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
*k&call_and_return_all_conditional_losses
l__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 45}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45]}}
?

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
*m&call_and_return_all_conditional_losses
n__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?mPmQmRmS!mT"mU'mV(mWvXvYvZv[!v\"v]'v^(v_"
	optimizer
X
0
1
2
3
!4
"5
'6
(7"
trackable_list_wrapper
X
0
1
2
3
!4
"5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-layer_metrics
.layer_regularization_losses
/non_trainable_variables
	variables

0layers
	trainable_variables
1metrics

regularization_losses
a__call__
b_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
oserving_default"
signature_map
:d-2dense1/kernel
:-2dense1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
2layer_metrics
3non_trainable_variables

4layers
	variables
regularization_losses
trainable_variables
5metrics
6layer_regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7layer_metrics
8non_trainable_variables

9layers
	variables
regularization_losses
trainable_variables
:metrics
;layer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:--2dense2/kernel
:-2dense2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
<layer_metrics
=non_trainable_variables

>layers
	variables
regularization_losses
trainable_variables
?metrics
@layer_regularization_losses
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Alayer_metrics
Bnon_trainable_variables

Clayers
	variables
regularization_losses
trainable_variables
Dmetrics
Elayer_regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 :-2encoder/kernel
:2encoder/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
Flayer_metrics
Gnon_trainable_variables

Hlayers
#	variables
$regularization_losses
%trainable_variables
Imetrics
Jlayer_regularization_losses
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
Klayer_metrics
Lnon_trainable_variables

Mlayers
)	variables
*regularization_losses
+trainable_variables
Nmetrics
Olayer_regularization_losses
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:d-2dense1/kernel/m
:-2dense1/bias/m
:--2dense2/kernel/m
:-2dense2/bias/m
 :-2encoder/kernel/m
:2encoder/bias/m
:2dense/kernel/m
:2dense/bias/m
:d-2dense1/kernel/v
:-2dense1/bias/v
:--2dense2/kernel/v
:-2dense2/bias/v
 :-2encoder/kernel/v
:2encoder/bias/v
:2dense/kernel/v
:2dense/bias/v
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_41401
E__inference_sequential_layer_call_and_return_conditional_losses_41433
E__inference_sequential_layer_call_and_return_conditional_losses_41237
E__inference_sequential_layer_call_and_return_conditional_losses_41211?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_41285
*__inference_sequential_layer_call_fn_41332
*__inference_sequential_layer_call_fn_41475
*__inference_sequential_layer_call_fn_41454?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_41040?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
dense1_input?????????d
?2?
A__inference_dense1_layer_call_and_return_conditional_losses_41486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense1_layer_call_fn_41495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout1_layer_call_and_return_conditional_losses_41507
C__inference_dropout1_layer_call_and_return_conditional_losses_41512?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout1_layer_call_fn_41522
(__inference_dropout1_layer_call_fn_41517?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense2_layer_call_and_return_conditional_losses_41533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense2_layer_call_fn_41542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout2_layer_call_and_return_conditional_losses_41554
C__inference_dropout2_layer_call_and_return_conditional_losses_41559?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout2_layer_call_fn_41564
(__inference_dropout2_layer_call_fn_41569?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_encoder_layer_call_and_return_conditional_losses_41579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_encoder_layer_call_fn_41588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_41598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_41607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_41355dense1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_41040p!"'(5?2
+?(
&?#
dense1_input?????????d
? "-?*
(
dense?
dense??????????
A__inference_dense1_layer_call_and_return_conditional_losses_41486\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????-
? y
&__inference_dense1_layer_call_fn_41495O/?,
%?"
 ?
inputs?????????d
? "??????????-?
A__inference_dense2_layer_call_and_return_conditional_losses_41533\/?,
%?"
 ?
inputs?????????-
? "%?"
?
0?????????-
? y
&__inference_dense2_layer_call_fn_41542O/?,
%?"
 ?
inputs?????????-
? "??????????-?
@__inference_dense_layer_call_and_return_conditional_losses_41598\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_41607O'(/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dropout1_layer_call_and_return_conditional_losses_41507\3?0
)?&
 ?
inputs?????????-
p
? "%?"
?
0?????????-
? ?
C__inference_dropout1_layer_call_and_return_conditional_losses_41512\3?0
)?&
 ?
inputs?????????-
p 
? "%?"
?
0?????????-
? {
(__inference_dropout1_layer_call_fn_41517O3?0
)?&
 ?
inputs?????????-
p
? "??????????-{
(__inference_dropout1_layer_call_fn_41522O3?0
)?&
 ?
inputs?????????-
p 
? "??????????-?
C__inference_dropout2_layer_call_and_return_conditional_losses_41554\3?0
)?&
 ?
inputs?????????-
p
? "%?"
?
0?????????-
? ?
C__inference_dropout2_layer_call_and_return_conditional_losses_41559\3?0
)?&
 ?
inputs?????????-
p 
? "%?"
?
0?????????-
? {
(__inference_dropout2_layer_call_fn_41564O3?0
)?&
 ?
inputs?????????-
p
? "??????????-{
(__inference_dropout2_layer_call_fn_41569O3?0
)?&
 ?
inputs?????????-
p 
? "??????????-?
B__inference_encoder_layer_call_and_return_conditional_losses_41579\!"/?,
%?"
 ?
inputs?????????-
? "%?"
?
0?????????
? z
'__inference_encoder_layer_call_fn_41588O!"/?,
%?"
 ?
inputs?????????-
? "???????????
E__inference_sequential_layer_call_and_return_conditional_losses_41211p!"'(=?:
3?0
&?#
dense1_input?????????d
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_41237p!"'(=?:
3?0
&?#
dense1_input?????????d
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_41401j!"'(7?4
-?*
 ?
inputs?????????d
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_41433j!"'(7?4
-?*
 ?
inputs?????????d
p 

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_41285c!"'(=?:
3?0
&?#
dense1_input?????????d
p

 
? "???????????
*__inference_sequential_layer_call_fn_41332c!"'(=?:
3?0
&?#
dense1_input?????????d
p 

 
? "???????????
*__inference_sequential_layer_call_fn_41454]!"'(7?4
-?*
 ?
inputs?????????d
p

 
? "???????????
*__inference_sequential_layer_call_fn_41475]!"'(7?4
-?*
 ?
inputs?????????d
p 

 
? "???????????
#__inference_signature_wrapper_41355?!"'(E?B
? 
;?8
6
dense1_input&?#
dense1_input?????????d"-?*
(
dense?
dense?????????