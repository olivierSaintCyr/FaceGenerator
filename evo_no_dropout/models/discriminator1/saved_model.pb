Ū
Ķ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18	

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
:@*
dtype0

instance_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!instance_normalization_10/gamma

3instance_normalization_10/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_10/gamma*
_output_shapes	
:*
dtype0

instance_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_10/beta

2instance_normalization_10/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_10/beta*
_output_shapes	
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
į*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
į*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

sequential_7/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namesequential_7/conv2d_4/kernel

0sequential_7/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpsequential_7/conv2d_4/kernel*&
_output_shapes
:*
dtype0

sequential_8/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namesequential_8/conv2d_5/kernel

0sequential_8/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpsequential_8/conv2d_5/kernel*&
_output_shapes
: *
dtype0

sequential_9/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_namesequential_9/conv2d_6/kernel

0sequential_9/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpsequential_9/conv2d_6/kernel*&
_output_shapes
: @*
dtype0

NoOpNoOp
é1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¤1
value1B1 B1

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 

layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api

layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api

layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
^

&kernel
'regularization_losses
(	variables
)trainable_variables
*	keras_api
g
	+gamma
,beta
-regularization_losses
.	variables
/trainable_variables
0	keras_api
R
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
 
8
?0
@1
A2
&3
+4
,5
96
:7
8
?0
@1
A2
&3
+4
,5
96
:7
­
Bnon_trainable_variables

Clayers
regularization_losses
	variables
trainable_variables
Dmetrics
Elayer_regularization_losses
Flayer_metrics
 
r
G_inbound_nodes

?kernel
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
f
L_inbound_nodes
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
 

?0

?0
­
Qnon_trainable_variables

Rlayers
regularization_losses
	variables
trainable_variables
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
r
V_inbound_nodes

@kernel
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
f
[_inbound_nodes
\regularization_losses
]	variables
^trainable_variables
_	keras_api
 

@0

@0
­
`non_trainable_variables

alayers
regularization_losses
	variables
trainable_variables
bmetrics
clayer_regularization_losses
dlayer_metrics
r
e_inbound_nodes

Akernel
fregularization_losses
g	variables
htrainable_variables
i	keras_api
f
j_inbound_nodes
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
 

A0

A0
­
onon_trainable_variables

players
regularization_losses
	variables
 trainable_variables
qmetrics
rlayer_regularization_losses
slayer_metrics
 
 
 
­
tnon_trainable_variables

ulayers
"regularization_losses
#	variables
$trainable_variables
vmetrics
wlayer_regularization_losses
xlayer_metrics
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
­
ynon_trainable_variables

zlayers
'regularization_losses
(	variables
)trainable_variables
{metrics
|layer_regularization_losses
}layer_metrics
jh
VARIABLE_VALUEinstance_normalization_10/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEinstance_normalization_10/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
°
~non_trainable_variables

layers
-regularization_losses
.	variables
/trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
²
non_trainable_variables
layers
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
²
non_trainable_variables
layers
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
layer_metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
²
non_trainable_variables
layers
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
layer_metrics
XV
VARIABLE_VALUEsequential_7/conv2d_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_8/conv2d_5/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_9/conv2d_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 

?0

?0
²
non_trainable_variables
layers
Hregularization_losses
I	variables
Jtrainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
 
²
non_trainable_variables
layers
Mregularization_losses
N	variables
Otrainable_variables
metrics
 layer_regularization_losses
layer_metrics
 

0
1
 
 
 
 
 

@0

@0
²
non_trainable_variables
layers
Wregularization_losses
X	variables
Ytrainable_variables
metrics
 layer_regularization_losses
 layer_metrics
 
 
 
 
²
”non_trainable_variables
¢layers
\regularization_losses
]	variables
^trainable_variables
£metrics
 ¤layer_regularization_losses
„layer_metrics
 

0
1
 
 
 
 
 

A0

A0
²
¦non_trainable_variables
§layers
fregularization_losses
g	variables
htrainable_variables
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
 
 
 
 
²
«non_trainable_variables
¬layers
kregularization_losses
l	variables
mtrainable_variables
­metrics
 ®layer_regularization_losses
Ælayer_metrics
 

0
1
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
 

serving_default_discriminatorPlaceholder*1
_output_shapes
:’’’’’’’’’*
dtype0*&
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_discriminatorsequential_7/conv2d_4/kernelsequential_8/conv2d_5/kernelsequential_9/conv2d_6/kernelconv2d_7/kernelinstance_normalization_10/gammainstance_normalization_10/betadense_2/kerneldense_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_10317
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_7/kernel/Read/ReadVariableOp3instance_normalization_10/gamma/Read/ReadVariableOp2instance_normalization_10/beta/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp0sequential_7/conv2d_4/kernel/Read/ReadVariableOp0sequential_8/conv2d_5/kernel/Read/ReadVariableOp0sequential_9/conv2d_6/kernel/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_10760
ź
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_7/kernelinstance_normalization_10/gammainstance_normalization_10/betadense_2/kerneldense_2/biassequential_7/conv2d_4/kernelsequential_8/conv2d_5/kernelsequential_9/conv2d_6/kernel*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_10794ĶĮ
ą

C__inference_conv2d_5_layer_call_and_return_conditional_losses_10672

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@::W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ä

C__inference_conv2d_4_layer_call_and_return_conditional_losses_10648

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ

£
F__inference_sequential_8_layer_call_and_return_conditional_losses_9848
conv2d_5_input
conv2d_5_9831
identity¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_9831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_98222"
 conv2d_5/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_98392
leaky_re_lu_7/PartitionedCall„
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’@@
(
_user_specified_nameconv2d_5_input
ń

£
F__inference_sequential_7_layer_call_and_return_conditional_losses_9780
conv2d_4_input
conv2d_4_9775
identity¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_9775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97462"
 conv2d_4/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_97632
leaky_re_lu_6/PartitionedCall„
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:a ]
1
_output_shapes
:’’’’’’’’’
(
_user_specified_nameconv2d_4_input
×

G__inference_sequential_8_layer_call_and_return_conditional_losses_10543

inputs+
'conv2d_5_conv2d_readvariableop_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp¾
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
conv2d_5/Conv2D
leaky_re_lu_7/LeakyRelu	LeakyReluconv2d_5/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’   *
alpha%>2
leaky_re_lu_7/LeakyRelu
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@::W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
õ
d
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_10606

inputs
identitym
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’*
alpha%>2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
c
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_9763

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@@:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
×

G__inference_sequential_9_layer_call_and_return_conditional_losses_10573

inputs+
'conv2d_6_conv2d_readvariableop_resource
identity°
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp¾
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
conv2d_6/Conv2D
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_6/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@*
alpha%>2
leaky_re_lu_8/LeakyRelu
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   ::W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
Ü
n
(__inference_conv2d_6_layer_call_fn_10703

inputs
unknown
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_98982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
Ł


F__inference_sequential_7_layer_call_and_return_conditional_losses_9791

inputs
conv2d_4_9786
identity¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_9786*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97462"
 conv2d_4/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_97632
leaky_re_lu_6/PartitionedCall„
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ

£
F__inference_sequential_8_layer_call_and_return_conditional_losses_9856
conv2d_5_input
conv2d_5_9851
identity¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_9851*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_98222"
 conv2d_5/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_98392
leaky_re_lu_7/PartitionedCall„
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’@@
(
_user_specified_nameconv2d_5_input
ū
y
+__inference_sequential_9_layer_call_fn_9948
conv2d_6_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’   
(
_user_specified_nameconv2d_6_input
Õ


F__inference_sequential_8_layer_call_and_return_conditional_losses_9882

inputs
conv2d_5_9877
identity¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_9877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_98222"
 conv2d_5/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_98392
leaky_re_lu_7/PartitionedCall„
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ū
y
+__inference_sequential_8_layer_call_fn_9887
conv2d_5_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’@@
(
_user_specified_nameconv2d_5_input
ä
r
,__inference_sequential_9_layer_call_fn_10580

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
Ń
Ŗ
B__inference_dense_2_layer_call_and_return_conditional_losses_10632

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
į*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*0
_input_shapes
:’’’’’’’’’į:::Q M
)
_output_shapes
:’’’’’’’’’į
 
_user_specified_nameinputs
¼
I
-__inference_leaky_re_lu_6_layer_call_fn_10665

inputs
identityŠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_97632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@@:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
+

>__inference_Dis_layer_call_and_return_conditional_losses_10275

inputs
sequential_7_10249
sequential_8_10252
sequential_9_10255
conv2d_7_10259#
instance_normalization_10_10262#
instance_normalization_10_10264
dense_2_10269
dense_2_10271
identity¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢1instance_normalization_10/StatefulPartitionedCall¢$sequential_7/StatefulPartitionedCall¢$sequential_8/StatefulPartitionedCall¢$sequential_9/StatefulPartitionedCall
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_10249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_98062&
$sequential_7/StatefulPartitionedCallĄ
$sequential_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0sequential_8_10252*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98822&
$sequential_8/StatefulPartitionedCallĄ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_10255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99582&
$sequential_9/StatefulPartitionedCall
 zero_padding2d_1/PartitionedCallPartitionedCall-sequential_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_99702"
 zero_padding2d_1/PartitionedCall®
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_7_10259*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_100932"
 conv2d_7/StatefulPartitionedCall
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0instance_normalization_10_10262instance_normalization_10_10264*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1002023
1instance_normalization_10/StatefulPartitionedCall¢
leaky_re_lu_9/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_101152
leaky_re_lu_9/PartitionedCallū
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’į* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_101292
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_10269dense_2_10271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101472!
dense_2/StatefulPartitionedCallź
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß

B__inference_conv2d_5_layer_call_and_return_conditional_losses_9822

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@::W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
+

>__inference_Dis_layer_call_and_return_conditional_losses_10164
discriminator
sequential_7_10048
sequential_8_10065
sequential_9_10082
conv2d_7_10102#
instance_normalization_10_10105#
instance_normalization_10_10107
dense_2_10158
dense_2_10160
identity¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢1instance_normalization_10/StatefulPartitionedCall¢$sequential_7/StatefulPartitionedCall¢$sequential_8/StatefulPartitionedCall¢$sequential_9/StatefulPartitionedCall 
$sequential_7/StatefulPartitionedCallStatefulPartitionedCalldiscriminatorsequential_7_10048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_97912&
$sequential_7/StatefulPartitionedCallĄ
$sequential_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0sequential_8_10065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98672&
$sequential_8/StatefulPartitionedCallĄ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_10082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99432&
$sequential_9/StatefulPartitionedCall
 zero_padding2d_1/PartitionedCallPartitionedCall-sequential_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_99702"
 zero_padding2d_1/PartitionedCall®
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_7_10102*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_100932"
 conv2d_7/StatefulPartitionedCall
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0instance_normalization_10_10105instance_normalization_10_10107*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1002023
1instance_normalization_10/StatefulPartitionedCall¢
leaky_re_lu_9/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_101152
leaky_re_lu_9/PartitionedCallū
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’į* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_101292
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_10158dense_2_10160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101472!
dense_2/StatefulPartitionedCallź
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
¼
I
-__inference_leaky_re_lu_7_layer_call_fn_10689

inputs
identityŠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_98392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’   :W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
ß

B__inference_conv2d_6_layer_call_and_return_conditional_losses_9898

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   ::W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
ä

C__inference_conv2d_7_layer_call_and_return_conditional_losses_10093

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp„
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
Conv2Dl
IdentityIdentityConv2D:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@::W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ł


F__inference_sequential_7_layer_call_and_return_conditional_losses_9806

inputs
conv2d_4_9801
identity¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_9801*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97462"
 conv2d_4/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_97632
leaky_re_lu_6/PartitionedCall„
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ć

B__inference_conv2d_4_layer_call_and_return_conditional_losses_9746

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
g

>__inference_Dis_layer_call_and_return_conditional_losses_10386

inputs8
4sequential_7_conv2d_4_conv2d_readvariableop_resource8
4sequential_8_conv2d_5_conv2d_readvariableop_resource8
4sequential_9_conv2d_6_conv2d_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource=
9instance_normalization_10_reshape_readvariableop_resource?
;instance_normalization_10_reshape_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity×
+sequential_7/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_7_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_7/conv2d_4/Conv2D/ReadVariableOpå
sequential_7/conv2d_4/Conv2DConv2Dinputs3sequential_7/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
sequential_7/conv2d_4/Conv2DĮ
$sequential_7/leaky_re_lu_6/LeakyRelu	LeakyRelu%sequential_7/conv2d_4/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2&
$sequential_7/leaky_re_lu_6/LeakyRelu×
+sequential_8/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_8/conv2d_5/Conv2D/ReadVariableOp
sequential_8/conv2d_5/Conv2DConv2D2sequential_7/leaky_re_lu_6/LeakyRelu:activations:03sequential_8/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
sequential_8/conv2d_5/Conv2DĮ
$sequential_8/leaky_re_lu_7/LeakyRelu	LeakyRelu%sequential_8/conv2d_5/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’   *
alpha%>2&
$sequential_8/leaky_re_lu_7/LeakyRelu×
+sequential_9/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_9_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_9/conv2d_6/Conv2D/ReadVariableOp
sequential_9/conv2d_6/Conv2DConv2D2sequential_8/leaky_re_lu_7/LeakyRelu:activations:03sequential_9/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
sequential_9/conv2d_6/Conv2DĮ
$sequential_9/leaky_re_lu_8/LeakyRelu	LeakyRelu%sequential_9/conv2d_6/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@*
alpha%>2&
$sequential_9/leaky_re_lu_8/LeakyReluÆ
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddingsÉ
zero_padding2d_1/PadPad2sequential_9/leaky_re_lu_8/LeakyRelu:activations:0&zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
zero_padding2d_1/Pad±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp×
conv2d_7/Conv2DConv2Dzero_padding2d_1/Pad:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
conv2d_7/Conv2D
instance_normalization_10/ShapeShapeconv2d_7/Conv2D:output:0*
T0*
_output_shapes
:2!
instance_normalization_10/ShapeØ
-instance_normalization_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-instance_normalization_10/strided_slice/stack¬
/instance_normalization_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice/stack_1¬
/instance_normalization_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice/stack_2ž
'instance_normalization_10/strided_sliceStridedSlice(instance_normalization_10/Shape:output:06instance_normalization_10/strided_slice/stack:output:08instance_normalization_10/strided_slice/stack_1:output:08instance_normalization_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'instance_normalization_10/strided_slice¬
/instance_normalization_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_1/stack°
1instance_normalization_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_1/stack_1°
1instance_normalization_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_1/stack_2
)instance_normalization_10/strided_slice_1StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_1/stack:output:0:instance_normalization_10/strided_slice_1/stack_1:output:0:instance_normalization_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_1¬
/instance_normalization_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_2/stack°
1instance_normalization_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_2/stack_1°
1instance_normalization_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_2/stack_2
)instance_normalization_10/strided_slice_2StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_2/stack:output:0:instance_normalization_10/strided_slice_2/stack_1:output:0:instance_normalization_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_2¬
/instance_normalization_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_3/stack°
1instance_normalization_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_3/stack_1°
1instance_normalization_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_3/stack_2
)instance_normalization_10/strided_slice_3StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_3/stack:output:0:instance_normalization_10/strided_slice_3/stack_1:output:0:instance_normalization_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_3Å
8instance_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2:
8instance_normalization_10/moments/mean/reduction_indices
&instance_normalization_10/moments/meanMeanconv2d_7/Conv2D:output:0Ainstance_normalization_10/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2(
&instance_normalization_10/moments/meanÜ
.instance_normalization_10/moments/StopGradientStopGradient/instance_normalization_10/moments/mean:output:0*
T0*0
_output_shapes
:’’’’’’’’’20
.instance_normalization_10/moments/StopGradient
3instance_normalization_10/moments/SquaredDifferenceSquaredDifferenceconv2d_7/Conv2D:output:07instance_normalization_10/moments/StopGradient:output:0*
T0*0
_output_shapes
:’’’’’’’’’25
3instance_normalization_10/moments/SquaredDifferenceĶ
<instance_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<instance_normalization_10/moments/variance/reduction_indices¬
*instance_normalization_10/moments/varianceMean7instance_normalization_10/moments/SquaredDifference:z:0Einstance_normalization_10/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2,
*instance_normalization_10/moments/varianceŪ
0instance_normalization_10/Reshape/ReadVariableOpReadVariableOp9instance_normalization_10_reshape_readvariableop_resource*
_output_shapes	
:*
dtype022
0instance_normalization_10/Reshape/ReadVariableOp«
'instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_10/Reshape/shapeļ
!instance_normalization_10/ReshapeReshape8instance_normalization_10/Reshape/ReadVariableOp:value:00instance_normalization_10/Reshape/shape:output:0*
T0*'
_output_shapes
:2#
!instance_normalization_10/Reshapeį
2instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype024
2instance_normalization_10/Reshape_1/ReadVariableOpÆ
)instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_10/Reshape_1/shape÷
#instance_normalization_10/Reshape_1Reshape:instance_normalization_10/Reshape_1/ReadVariableOp:value:02instance_normalization_10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:2%
#instance_normalization_10/Reshape_1
)instance_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)instance_normalization_10/batchnorm/add/y’
'instance_normalization_10/batchnorm/addAddV23instance_normalization_10/moments/variance:output:02instance_normalization_10/batchnorm/add/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/addĒ
)instance_normalization_10/batchnorm/RsqrtRsqrt+instance_normalization_10/batchnorm/add:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/Rsqrtļ
'instance_normalization_10/batchnorm/mulMul-instance_normalization_10/batchnorm/Rsqrt:y:0*instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/mulß
)instance_normalization_10/batchnorm/mul_1Mulconv2d_7/Conv2D:output:0+instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/mul_1ö
)instance_normalization_10/batchnorm/mul_2Mul/instance_normalization_10/moments/mean:output:0+instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/mul_2ń
'instance_normalization_10/batchnorm/subSub,instance_normalization_10/Reshape_1:output:0-instance_normalization_10/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/subö
)instance_normalization_10/batchnorm/add_1AddV2-instance_normalization_10/batchnorm/mul_1:z:0+instance_normalization_10/batchnorm/sub:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/add_1°
leaky_re_lu_9/LeakyRelu	LeakyRelu-instance_normalization_10/batchnorm/add_1:z:0*0
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_9/LeakyRelus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’p  2
flatten_1/Const¦
flatten_1/ReshapeReshape%leaky_re_lu_9/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2
flatten_1/Reshape§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
į*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’:::::::::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä

C__inference_conv2d_7_layer_call_and_return_conditional_losses_10594

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp„
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
Conv2Dl
IdentityIdentityConv2D:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@::W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
+

>__inference_Dis_layer_call_and_return_conditional_losses_10225

inputs
sequential_7_10199
sequential_8_10202
sequential_9_10205
conv2d_7_10209#
instance_normalization_10_10212#
instance_normalization_10_10214
dense_2_10219
dense_2_10221
identity¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢1instance_normalization_10/StatefulPartitionedCall¢$sequential_7/StatefulPartitionedCall¢$sequential_8/StatefulPartitionedCall¢$sequential_9/StatefulPartitionedCall
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_10199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_97912&
$sequential_7/StatefulPartitionedCallĄ
$sequential_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0sequential_8_10202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98672&
$sequential_8/StatefulPartitionedCallĄ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_10205*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99432&
$sequential_9/StatefulPartitionedCall
 zero_padding2d_1/PartitionedCallPartitionedCall-sequential_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_99702"
 zero_padding2d_1/PartitionedCall®
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_7_10209*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_100932"
 conv2d_7/StatefulPartitionedCall
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0instance_normalization_10_10212instance_normalization_10_10214*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1002023
1instance_normalization_10/StatefulPartitionedCall¢
leaky_re_lu_9/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_101152
leaky_re_lu_9/PartitionedCallū
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’į* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_101292
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_10219dense_2_10221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101472!
dense_2/StatefulPartitionedCallź
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
d
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_10115

inputs
identitym
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’*
alpha%>2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
“ 
­
__inference__traced_save_10760
file_prefix.
*savev2_conv2d_7_kernel_read_readvariableop>
:savev2_instance_normalization_10_gamma_read_readvariableop=
9savev2_instance_normalization_10_beta_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop;
7savev2_sequential_7_conv2d_4_kernel_read_readvariableop;
7savev2_sequential_8_conv2d_5_kernel_read_readvariableop;
7savev2_sequential_9_conv2d_6_kernel_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_be5f39c3c15a42b3aa241d3f5e8de0a2/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¬
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*¾
value“B±	B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesä
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_7_kernel_read_readvariableop:savev2_instance_normalization_10_gamma_read_readvariableop9savev2_instance_normalization_10_beta_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop7savev2_sequential_7_conv2d_4_kernel_read_readvariableop7savev2_sequential_8_conv2d_5_kernel_read_readvariableop7savev2_sequential_9_conv2d_6_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
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

identity_1Identity_1:output:0*
_input_shapeso
m: :@:::
į::: : @: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
į: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
: @:	

_output_shapes
: 
Ń
Ŗ
B__inference_dense_2_layer_call_and_return_conditional_losses_10147

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
į*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*0
_input_shapes
:’’’’’’’’’į:::Q M
)
_output_shapes
:’’’’’’’’’į
 
_user_specified_nameinputs
ń
d
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_10708

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’@*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
×

G__inference_sequential_9_layer_call_and_return_conditional_losses_10565

inputs+
'conv2d_6_conv2d_readvariableop_resource
identity°
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp¾
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
conv2d_6/Conv2D
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_6/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@*
alpha%>2
leaky_re_lu_8/LeakyRelu
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   ::W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
½
Ł
#__inference_Dis_layer_call_fn_10244
discriminator
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĘ
StatefulPartitionedCallStatefulPartitionedCalldiscriminatorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_Dis_layer_call_and_return_conditional_losses_102252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
+

>__inference_Dis_layer_call_and_return_conditional_losses_10193
discriminator
sequential_7_10167
sequential_8_10170
sequential_9_10173
conv2d_7_10177#
instance_normalization_10_10180#
instance_normalization_10_10182
dense_2_10187
dense_2_10189
identity¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢1instance_normalization_10/StatefulPartitionedCall¢$sequential_7/StatefulPartitionedCall¢$sequential_8/StatefulPartitionedCall¢$sequential_9/StatefulPartitionedCall 
$sequential_7/StatefulPartitionedCallStatefulPartitionedCalldiscriminatorsequential_7_10167*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_98062&
$sequential_7/StatefulPartitionedCallĄ
$sequential_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0sequential_8_10170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98822&
$sequential_8/StatefulPartitionedCallĄ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_10173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99582&
$sequential_9/StatefulPartitionedCall
 zero_padding2d_1/PartitionedCallPartitionedCall-sequential_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_99702"
 zero_padding2d_1/PartitionedCall®
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_7_10177*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_100932"
 conv2d_7/StatefulPartitionedCall
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0instance_normalization_10_10180instance_normalization_10_10182*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1002023
1instance_normalization_10/StatefulPartitionedCall¢
leaky_re_lu_9/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_101152
leaky_re_lu_9/PartitionedCallū
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’į* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_101292
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_10187dense_2_10189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101472!
dense_2/StatefulPartitionedCallź
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
č
r
,__inference_sequential_7_layer_call_fn_10527

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_98062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ū
y
+__inference_sequential_8_layer_call_fn_9872
conv2d_5_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’@@
(
_user_specified_nameconv2d_5_input
¼
I
-__inference_leaky_re_lu_8_layer_call_fn_10713

inputs
identityŠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_99152
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
m

__inference__wrapped_model_9735
discriminator<
8dis_sequential_7_conv2d_4_conv2d_readvariableop_resource<
8dis_sequential_8_conv2d_5_conv2d_readvariableop_resource<
8dis_sequential_9_conv2d_6_conv2d_readvariableop_resource/
+dis_conv2d_7_conv2d_readvariableop_resourceA
=dis_instance_normalization_10_reshape_readvariableop_resourceC
?dis_instance_normalization_10_reshape_1_readvariableop_resource.
*dis_dense_2_matmul_readvariableop_resource/
+dis_dense_2_biasadd_readvariableop_resource
identityć
/Dis/sequential_7/conv2d_4/Conv2D/ReadVariableOpReadVariableOp8dis_sequential_7_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/Dis/sequential_7/conv2d_4/Conv2D/ReadVariableOpų
 Dis/sequential_7/conv2d_4/Conv2DConv2Ddiscriminator7Dis/sequential_7/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2"
 Dis/sequential_7/conv2d_4/Conv2DĶ
(Dis/sequential_7/leaky_re_lu_6/LeakyRelu	LeakyRelu)Dis/sequential_7/conv2d_4/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2*
(Dis/sequential_7/leaky_re_lu_6/LeakyReluć
/Dis/sequential_8/conv2d_5/Conv2D/ReadVariableOpReadVariableOp8dis_sequential_8_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/Dis/sequential_8/conv2d_5/Conv2D/ReadVariableOp”
 Dis/sequential_8/conv2d_5/Conv2DConv2D6Dis/sequential_7/leaky_re_lu_6/LeakyRelu:activations:07Dis/sequential_8/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2"
 Dis/sequential_8/conv2d_5/Conv2DĶ
(Dis/sequential_8/leaky_re_lu_7/LeakyRelu	LeakyRelu)Dis/sequential_8/conv2d_5/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’   *
alpha%>2*
(Dis/sequential_8/leaky_re_lu_7/LeakyReluć
/Dis/sequential_9/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8dis_sequential_9_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/Dis/sequential_9/conv2d_6/Conv2D/ReadVariableOp”
 Dis/sequential_9/conv2d_6/Conv2DConv2D6Dis/sequential_8/leaky_re_lu_7/LeakyRelu:activations:07Dis/sequential_9/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2"
 Dis/sequential_9/conv2d_6/Conv2DĶ
(Dis/sequential_9/leaky_re_lu_8/LeakyRelu	LeakyRelu)Dis/sequential_9/conv2d_6/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@*
alpha%>2*
(Dis/sequential_9/leaky_re_lu_8/LeakyRelu·
!Dis/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2#
!Dis/zero_padding2d_1/Pad/paddingsŁ
Dis/zero_padding2d_1/PadPad6Dis/sequential_9/leaky_re_lu_8/LeakyRelu:activations:0*Dis/zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
Dis/zero_padding2d_1/Pad½
"Dis/conv2d_7/Conv2D/ReadVariableOpReadVariableOp+dis_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"Dis/conv2d_7/Conv2D/ReadVariableOpē
Dis/conv2d_7/Conv2DConv2D!Dis/zero_padding2d_1/Pad:output:0*Dis/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
Dis/conv2d_7/Conv2D
#Dis/instance_normalization_10/ShapeShapeDis/conv2d_7/Conv2D:output:0*
T0*
_output_shapes
:2%
#Dis/instance_normalization_10/Shape°
1Dis/instance_normalization_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1Dis/instance_normalization_10/strided_slice/stack“
3Dis/instance_normalization_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Dis/instance_normalization_10/strided_slice/stack_1“
3Dis/instance_normalization_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Dis/instance_normalization_10/strided_slice/stack_2
+Dis/instance_normalization_10/strided_sliceStridedSlice,Dis/instance_normalization_10/Shape:output:0:Dis/instance_normalization_10/strided_slice/stack:output:0<Dis/instance_normalization_10/strided_slice/stack_1:output:0<Dis/instance_normalization_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Dis/instance_normalization_10/strided_slice“
3Dis/instance_normalization_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Dis/instance_normalization_10/strided_slice_1/stackø
5Dis/instance_normalization_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_1/stack_1ø
5Dis/instance_normalization_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_1/stack_2 
-Dis/instance_normalization_10/strided_slice_1StridedSlice,Dis/instance_normalization_10/Shape:output:0<Dis/instance_normalization_10/strided_slice_1/stack:output:0>Dis/instance_normalization_10/strided_slice_1/stack_1:output:0>Dis/instance_normalization_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Dis/instance_normalization_10/strided_slice_1“
3Dis/instance_normalization_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Dis/instance_normalization_10/strided_slice_2/stackø
5Dis/instance_normalization_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_2/stack_1ø
5Dis/instance_normalization_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_2/stack_2 
-Dis/instance_normalization_10/strided_slice_2StridedSlice,Dis/instance_normalization_10/Shape:output:0<Dis/instance_normalization_10/strided_slice_2/stack:output:0>Dis/instance_normalization_10/strided_slice_2/stack_1:output:0>Dis/instance_normalization_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Dis/instance_normalization_10/strided_slice_2“
3Dis/instance_normalization_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Dis/instance_normalization_10/strided_slice_3/stackø
5Dis/instance_normalization_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_3/stack_1ø
5Dis/instance_normalization_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Dis/instance_normalization_10/strided_slice_3/stack_2 
-Dis/instance_normalization_10/strided_slice_3StridedSlice,Dis/instance_normalization_10/Shape:output:0<Dis/instance_normalization_10/strided_slice_3/stack:output:0>Dis/instance_normalization_10/strided_slice_3/stack_1:output:0>Dis/instance_normalization_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Dis/instance_normalization_10/strided_slice_3Ķ
<Dis/instance_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<Dis/instance_normalization_10/moments/mean/reduction_indices
*Dis/instance_normalization_10/moments/meanMeanDis/conv2d_7/Conv2D:output:0EDis/instance_normalization_10/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2,
*Dis/instance_normalization_10/moments/meanč
2Dis/instance_normalization_10/moments/StopGradientStopGradient3Dis/instance_normalization_10/moments/mean:output:0*
T0*0
_output_shapes
:’’’’’’’’’24
2Dis/instance_normalization_10/moments/StopGradient
7Dis/instance_normalization_10/moments/SquaredDifferenceSquaredDifferenceDis/conv2d_7/Conv2D:output:0;Dis/instance_normalization_10/moments/StopGradient:output:0*
T0*0
_output_shapes
:’’’’’’’’’29
7Dis/instance_normalization_10/moments/SquaredDifferenceÕ
@Dis/instance_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@Dis/instance_normalization_10/moments/variance/reduction_indices¼
.Dis/instance_normalization_10/moments/varianceMean;Dis/instance_normalization_10/moments/SquaredDifference:z:0IDis/instance_normalization_10/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(20
.Dis/instance_normalization_10/moments/varianceē
4Dis/instance_normalization_10/Reshape/ReadVariableOpReadVariableOp=dis_instance_normalization_10_reshape_readvariableop_resource*
_output_shapes	
:*
dtype026
4Dis/instance_normalization_10/Reshape/ReadVariableOp³
+Dis/instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2-
+Dis/instance_normalization_10/Reshape/shape’
%Dis/instance_normalization_10/ReshapeReshape<Dis/instance_normalization_10/Reshape/ReadVariableOp:value:04Dis/instance_normalization_10/Reshape/shape:output:0*
T0*'
_output_shapes
:2'
%Dis/instance_normalization_10/Reshapeķ
6Dis/instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOp?dis_instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype028
6Dis/instance_normalization_10/Reshape_1/ReadVariableOp·
-Dis/instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2/
-Dis/instance_normalization_10/Reshape_1/shape
'Dis/instance_normalization_10/Reshape_1Reshape>Dis/instance_normalization_10/Reshape_1/ReadVariableOp:value:06Dis/instance_normalization_10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:2)
'Dis/instance_normalization_10/Reshape_1£
-Dis/instance_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Dis/instance_normalization_10/batchnorm/add/y
+Dis/instance_normalization_10/batchnorm/addAddV27Dis/instance_normalization_10/moments/variance:output:06Dis/instance_normalization_10/batchnorm/add/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2-
+Dis/instance_normalization_10/batchnorm/addÓ
-Dis/instance_normalization_10/batchnorm/RsqrtRsqrt/Dis/instance_normalization_10/batchnorm/add:z:0*
T0*0
_output_shapes
:’’’’’’’’’2/
-Dis/instance_normalization_10/batchnorm/Rsqrt’
+Dis/instance_normalization_10/batchnorm/mulMul1Dis/instance_normalization_10/batchnorm/Rsqrt:y:0.Dis/instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:’’’’’’’’’2-
+Dis/instance_normalization_10/batchnorm/mulļ
-Dis/instance_normalization_10/batchnorm/mul_1MulDis/conv2d_7/Conv2D:output:0/Dis/instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2/
-Dis/instance_normalization_10/batchnorm/mul_1
-Dis/instance_normalization_10/batchnorm/mul_2Mul3Dis/instance_normalization_10/moments/mean:output:0/Dis/instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2/
-Dis/instance_normalization_10/batchnorm/mul_2
+Dis/instance_normalization_10/batchnorm/subSub0Dis/instance_normalization_10/Reshape_1:output:01Dis/instance_normalization_10/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:’’’’’’’’’2-
+Dis/instance_normalization_10/batchnorm/sub
-Dis/instance_normalization_10/batchnorm/add_1AddV21Dis/instance_normalization_10/batchnorm/mul_1:z:0/Dis/instance_normalization_10/batchnorm/sub:z:0*
T0*0
_output_shapes
:’’’’’’’’’2/
-Dis/instance_normalization_10/batchnorm/add_1¼
Dis/leaky_re_lu_9/LeakyRelu	LeakyRelu1Dis/instance_normalization_10/batchnorm/add_1:z:0*0
_output_shapes
:’’’’’’’’’*
alpha%>2
Dis/leaky_re_lu_9/LeakyRelu{
Dis/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’p  2
Dis/flatten_1/Const¶
Dis/flatten_1/ReshapeReshape)Dis/leaky_re_lu_9/LeakyRelu:activations:0Dis/flatten_1/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2
Dis/flatten_1/Reshape³
!Dis/dense_2/MatMul/ReadVariableOpReadVariableOp*dis_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
į*
dtype02#
!Dis/dense_2/MatMul/ReadVariableOpÆ
Dis/dense_2/MatMulMatMulDis/flatten_1/Reshape:output:0)Dis/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Dis/dense_2/MatMul°
"Dis/dense_2/BiasAdd/ReadVariableOpReadVariableOp+dis_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"Dis/dense_2/BiasAdd/ReadVariableOp±
Dis/dense_2/BiasAddBiasAddDis/dense_2/MatMul:product:0*Dis/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Dis/dense_2/BiasAddp
IdentityIdentityDis/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’:::::::::` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
Ø
Ņ
#__inference_Dis_layer_call_fn_10476

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_Dis_layer_call_and_return_conditional_losses_102252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū

G__inference_sequential_7_layer_call_and_return_conditional_losses_10513

inputs+
'conv2d_4_conv2d_readvariableop_resource
identity°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¾
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
conv2d_4/Conv2D
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_4/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2
leaky_re_lu_6/LeakyRelu
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
y
+__inference_sequential_7_layer_call_fn_9811
conv2d_4_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_98062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:’’’’’’’’’
(
_user_specified_nameconv2d_4_input
g

>__inference_Dis_layer_call_and_return_conditional_losses_10455

inputs8
4sequential_7_conv2d_4_conv2d_readvariableop_resource8
4sequential_8_conv2d_5_conv2d_readvariableop_resource8
4sequential_9_conv2d_6_conv2d_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource=
9instance_normalization_10_reshape_readvariableop_resource?
;instance_normalization_10_reshape_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity×
+sequential_7/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_7_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_7/conv2d_4/Conv2D/ReadVariableOpå
sequential_7/conv2d_4/Conv2DConv2Dinputs3sequential_7/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
sequential_7/conv2d_4/Conv2DĮ
$sequential_7/leaky_re_lu_6/LeakyRelu	LeakyRelu%sequential_7/conv2d_4/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2&
$sequential_7/leaky_re_lu_6/LeakyRelu×
+sequential_8/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_8/conv2d_5/Conv2D/ReadVariableOp
sequential_8/conv2d_5/Conv2DConv2D2sequential_7/leaky_re_lu_6/LeakyRelu:activations:03sequential_8/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
sequential_8/conv2d_5/Conv2DĮ
$sequential_8/leaky_re_lu_7/LeakyRelu	LeakyRelu%sequential_8/conv2d_5/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’   *
alpha%>2&
$sequential_8/leaky_re_lu_7/LeakyRelu×
+sequential_9/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_9_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_9/conv2d_6/Conv2D/ReadVariableOp
sequential_9/conv2d_6/Conv2DConv2D2sequential_8/leaky_re_lu_7/LeakyRelu:activations:03sequential_9/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
sequential_9/conv2d_6/Conv2DĮ
$sequential_9/leaky_re_lu_8/LeakyRelu	LeakyRelu%sequential_9/conv2d_6/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@*
alpha%>2&
$sequential_9/leaky_re_lu_8/LeakyReluÆ
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddingsÉ
zero_padding2d_1/PadPad2sequential_9/leaky_re_lu_8/LeakyRelu:activations:0&zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
zero_padding2d_1/Pad±
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp×
conv2d_7/Conv2DConv2Dzero_padding2d_1/Pad:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
conv2d_7/Conv2D
instance_normalization_10/ShapeShapeconv2d_7/Conv2D:output:0*
T0*
_output_shapes
:2!
instance_normalization_10/ShapeØ
-instance_normalization_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-instance_normalization_10/strided_slice/stack¬
/instance_normalization_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice/stack_1¬
/instance_normalization_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice/stack_2ž
'instance_normalization_10/strided_sliceStridedSlice(instance_normalization_10/Shape:output:06instance_normalization_10/strided_slice/stack:output:08instance_normalization_10/strided_slice/stack_1:output:08instance_normalization_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'instance_normalization_10/strided_slice¬
/instance_normalization_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_1/stack°
1instance_normalization_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_1/stack_1°
1instance_normalization_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_1/stack_2
)instance_normalization_10/strided_slice_1StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_1/stack:output:0:instance_normalization_10/strided_slice_1/stack_1:output:0:instance_normalization_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_1¬
/instance_normalization_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_2/stack°
1instance_normalization_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_2/stack_1°
1instance_normalization_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_2/stack_2
)instance_normalization_10/strided_slice_2StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_2/stack:output:0:instance_normalization_10/strided_slice_2/stack_1:output:0:instance_normalization_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_2¬
/instance_normalization_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/instance_normalization_10/strided_slice_3/stack°
1instance_normalization_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_3/stack_1°
1instance_normalization_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1instance_normalization_10/strided_slice_3/stack_2
)instance_normalization_10/strided_slice_3StridedSlice(instance_normalization_10/Shape:output:08instance_normalization_10/strided_slice_3/stack:output:0:instance_normalization_10/strided_slice_3/stack_1:output:0:instance_normalization_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)instance_normalization_10/strided_slice_3Å
8instance_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2:
8instance_normalization_10/moments/mean/reduction_indices
&instance_normalization_10/moments/meanMeanconv2d_7/Conv2D:output:0Ainstance_normalization_10/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2(
&instance_normalization_10/moments/meanÜ
.instance_normalization_10/moments/StopGradientStopGradient/instance_normalization_10/moments/mean:output:0*
T0*0
_output_shapes
:’’’’’’’’’20
.instance_normalization_10/moments/StopGradient
3instance_normalization_10/moments/SquaredDifferenceSquaredDifferenceconv2d_7/Conv2D:output:07instance_normalization_10/moments/StopGradient:output:0*
T0*0
_output_shapes
:’’’’’’’’’25
3instance_normalization_10/moments/SquaredDifferenceĶ
<instance_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<instance_normalization_10/moments/variance/reduction_indices¬
*instance_normalization_10/moments/varianceMean7instance_normalization_10/moments/SquaredDifference:z:0Einstance_normalization_10/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2,
*instance_normalization_10/moments/varianceŪ
0instance_normalization_10/Reshape/ReadVariableOpReadVariableOp9instance_normalization_10_reshape_readvariableop_resource*
_output_shapes	
:*
dtype022
0instance_normalization_10/Reshape/ReadVariableOp«
'instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_10/Reshape/shapeļ
!instance_normalization_10/ReshapeReshape8instance_normalization_10/Reshape/ReadVariableOp:value:00instance_normalization_10/Reshape/shape:output:0*
T0*'
_output_shapes
:2#
!instance_normalization_10/Reshapeį
2instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype024
2instance_normalization_10/Reshape_1/ReadVariableOpÆ
)instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_10/Reshape_1/shape÷
#instance_normalization_10/Reshape_1Reshape:instance_normalization_10/Reshape_1/ReadVariableOp:value:02instance_normalization_10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:2%
#instance_normalization_10/Reshape_1
)instance_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)instance_normalization_10/batchnorm/add/y’
'instance_normalization_10/batchnorm/addAddV23instance_normalization_10/moments/variance:output:02instance_normalization_10/batchnorm/add/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/addĒ
)instance_normalization_10/batchnorm/RsqrtRsqrt+instance_normalization_10/batchnorm/add:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/Rsqrtļ
'instance_normalization_10/batchnorm/mulMul-instance_normalization_10/batchnorm/Rsqrt:y:0*instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/mulß
)instance_normalization_10/batchnorm/mul_1Mulconv2d_7/Conv2D:output:0+instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/mul_1ö
)instance_normalization_10/batchnorm/mul_2Mul/instance_normalization_10/moments/mean:output:0+instance_normalization_10/batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/mul_2ń
'instance_normalization_10/batchnorm/subSub,instance_normalization_10/Reshape_1:output:0-instance_normalization_10/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:’’’’’’’’’2)
'instance_normalization_10/batchnorm/subö
)instance_normalization_10/batchnorm/add_1AddV2-instance_normalization_10/batchnorm/mul_1:z:0+instance_normalization_10/batchnorm/sub:z:0*
T0*0
_output_shapes
:’’’’’’’’’2+
)instance_normalization_10/batchnorm/add_1°
leaky_re_lu_9/LeakyRelu	LeakyRelu-instance_normalization_10/batchnorm/add_1:z:0*0
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_9/LeakyRelus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’p  2
flatten_1/Const¦
flatten_1/ReshapeReshape%leaky_re_lu_9/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2
flatten_1/Reshape§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
į*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’:::::::::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß
n
(__inference_conv2d_7_layer_call_fn_10601

inputs
unknown
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_100932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ü
n
(__inference_conv2d_5_layer_call_fn_10679

inputs
unknown
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_98222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
½
Ł
#__inference_Dis_layer_call_fn_10294
discriminator
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĘ
StatefulPartitionedCallStatefulPartitionedCalldiscriminatorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_Dis_layer_call_and_return_conditional_losses_102752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
ą

C__inference_conv2d_6_layer_call_and_return_conditional_losses_10696

inputs"
conv2d_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   ::W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
®
K
/__inference_zero_padding2d_1_layer_call_fn_9976

inputs
identityī
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_99702
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ą
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_10129

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’p  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
y
+__inference_sequential_7_layer_call_fn_9796
conv2d_4_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_97912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:’’’’’’’’’
(
_user_specified_nameconv2d_4_input
ū
y
+__inference_sequential_9_layer_call_fn_9963
conv2d_6_input
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’   
(
_user_specified_nameconv2d_6_input
×

G__inference_sequential_8_layer_call_and_return_conditional_losses_10535

inputs+
'conv2d_5_conv2d_readvariableop_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp¾
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’   *
paddingSAME*
strides
2
conv2d_5/Conv2D
leaky_re_lu_7/LeakyRelu	LeakyReluconv2d_5/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’   *
alpha%>2
leaky_re_lu_7/LeakyRelu
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@::W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ä
r
,__inference_sequential_8_layer_call_fn_10550

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ä
r
,__inference_sequential_9_layer_call_fn_10587

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_9_layer_call_and_return_conditional_losses_99582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
ć.
æ
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_10020

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ģ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ģ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ģ
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2 
moments/mean/reduction_indices”
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
moments/StopGradientæ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"moments/variance/reduction_indicesÄ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
	keep_dims(2
moments/variance
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:2	
Reshape
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:2
	Reshape_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
batchnorm/addy
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:’’’’’’’’’2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:’’’’’’’’’2
batchnorm/mul_2
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:’’’’’’’’’2
batchnorm/sub 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
š
c
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_9839

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’   *
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’   :W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
«
E
)__inference_flatten_1_layer_call_fn_10622

inputs
identityĒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’į* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_101292
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū

G__inference_sequential_7_layer_call_and_return_conditional_losses_10505

inputs+
'conv2d_4_conv2d_readvariableop_resource
identity°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¾
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
2
conv2d_4/Conv2D
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_4/Conv2D:output:0*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2
leaky_re_lu_6/LeakyRelu
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’::Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
n
(__inference_conv2d_4_layer_call_fn_10655

inputs
unknown
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č
r
,__inference_sequential_7_layer_call_fn_10520

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_97912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ


F__inference_sequential_9_layer_call_and_return_conditional_losses_9943

inputs
conv2d_6_9938
identity¢ conv2d_6/StatefulPartitionedCall
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_9938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_98982"
 conv2d_6/StatefulPartitionedCall
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_99152
leaky_re_lu_8/PartitionedCall„
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
ģ

9__inference_instance_normalization_10_layer_call_fn_10030

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_100202
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ė&
ķ
!__inference__traced_restore_10794
file_prefix$
 assignvariableop_conv2d_7_kernel6
2assignvariableop_1_instance_normalization_10_gamma5
1assignvariableop_2_instance_normalization_10_beta%
!assignvariableop_3_dense_2_kernel#
assignvariableop_4_dense_2_bias3
/assignvariableop_5_sequential_7_conv2d_4_kernel3
/assignvariableop_6_sequential_8_conv2d_5_kernel3
/assignvariableop_7_sequential_9_conv2d_6_kernel

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7²
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*¾
value“B±	B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesŲ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1·
AssignVariableOp_1AssignVariableOp2assignvariableop_1_instance_normalization_10_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp1assignvariableop_2_instance_normalization_10_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_2_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5“
AssignVariableOp_5AssignVariableOp/assignvariableop_5_sequential_7_conv2d_4_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6“
AssignVariableOp_6AssignVariableOp/assignvariableop_6_sequential_8_conv2d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7“
AssignVariableOp_7AssignVariableOp/assignvariableop_7_sequential_9_conv2d_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Ł
#__inference_signature_wrapper_10317
discriminator
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalldiscriminatorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_97352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:’’’’’’’’’
'
_user_specified_namediscriminator
Ą
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_10617

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’p  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:’’’’’’’’’į2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä
f
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_9970

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ä
r
,__inference_sequential_8_layer_call_fn_10557

inputs
unknown
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_8_layer_call_and_return_conditional_losses_98822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ß
|
'__inference_dense_2_layer_call_fn_10641

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*0
_input_shapes
:’’’’’’’’’į::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:’’’’’’’’’į
 
_user_specified_nameinputs
Õ


F__inference_sequential_9_layer_call_and_return_conditional_losses_9958

inputs
conv2d_6_9953
identity¢ conv2d_6/StatefulPartitionedCall
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_9953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_98982"
 conv2d_6/StatefulPartitionedCall
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_99152
leaky_re_lu_8/PartitionedCall„
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs
ķ

£
F__inference_sequential_9_layer_call_and_return_conditional_losses_9932
conv2d_6_input
conv2d_6_9927
identity¢ conv2d_6/StatefulPartitionedCall
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_9927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_98982"
 conv2d_6/StatefulPartitionedCall
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_99152
leaky_re_lu_8/PartitionedCall„
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’   
(
_user_specified_nameconv2d_6_input
ń
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_10660

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’@@*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@@:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ń

£
F__inference_sequential_7_layer_call_and_return_conditional_losses_9772
conv2d_4_input
conv2d_4_9755
identity¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_9755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97462"
 conv2d_4/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_97632
leaky_re_lu_6/PartitionedCall„
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’:2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:a ]
1
_output_shapes
:’’’’’’’’’
(
_user_specified_nameconv2d_4_input
ķ

£
F__inference_sequential_9_layer_call_and_return_conditional_losses_9924
conv2d_6_input
conv2d_6_9907
identity¢ conv2d_6/StatefulPartitionedCall
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_9907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_98982"
 conv2d_6/StatefulPartitionedCall
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_99152
leaky_re_lu_8/PartitionedCall„
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’   :2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’   
(
_user_specified_nameconv2d_6_input
Į
I
-__inference_leaky_re_lu_9_layer_call_fn_10611

inputs
identityŅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_101152
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
c
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_9915

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’@*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Õ


F__inference_sequential_8_layer_call_and_return_conditional_losses_9867

inputs
conv2d_5_9862
identity¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_9862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_98222"
 conv2d_5/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_98392
leaky_re_lu_7/PartitionedCall„
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’@@:2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
Ø
Ņ
#__inference_Dis_layer_call_fn_10497

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_Dis_layer_call_and_return_conditional_losses_102752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ń
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_10684

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:’’’’’’’’’   *
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’   2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’   :W S
/
_output_shapes
:’’’’’’’’’   
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ą
serving_default¬
Q
discriminator@
serving_default_discriminator:0’’’’’’’’’;
dense_20
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ōó
n
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
°__call__
+±&call_and_return_all_conditional_losses
²_default_save_signature"½j
_tf_keras_network”j{"class_name": "Functional", "name": "Dis", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Dis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "discriminator"}, "name": "discriminator", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_7", "inbound_nodes": [[["discriminator", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_8", "inbound_nodes": [[["sequential_7", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_9", "inbound_nodes": [[["sequential_8", 1, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["sequential_9", 1, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "groups": 128, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_10", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_9", "inbound_nodes": [[["instance_normalization_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["discriminator", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Dis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "discriminator"}, "name": "discriminator", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_7", "inbound_nodes": [[["discriminator", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_8", "inbound_nodes": [[["sequential_7", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "name": "sequential_9", "inbound_nodes": [[["sequential_8", 1, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["sequential_9", 1, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Addons>InstanceNormalization", "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "groups": 128, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_10", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_9", "inbound_nodes": [[["instance_normalization_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["discriminator", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
"
_tf_keras_input_layerę{"class_name": "InputLayer", "name": "discriminator", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "discriminator"}}

layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
³__call__
+“&call_and_return_all_conditional_losses"¼
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}}}
’
layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"ŗ
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}}}
’
layer_with_weights-0
layer-0
layer-1
regularization_losses
	variables
 trainable_variables
!	keras_api
·__call__
+ø&call_and_return_all_conditional_losses"ŗ
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}]}}}

"regularization_losses
#	variables
$trainable_variables
%	keras_api
+¹&call_and_return_all_conditional_losses
ŗ__call__"ś
_tf_keras_layerą{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



&kernel
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"ī
_tf_keras_layerŌ{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 64]}}
ö
	+gamma
,beta
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"Š
_tf_keras_layer¶{"class_name": "Addons>InstanceNormalization", "name": "instance_normalization_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "groups": 128, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 128]}}
ą
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+æ&call_and_return_all_conditional_losses
Ą__call__"Ļ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
č
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+Į&call_and_return_all_conditional_losses
Ā__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ł

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+Ć&call_and_return_all_conditional_losses
Ä__call__"Ņ
_tf_keras_layerø{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28800]}}
 "
trackable_list_wrapper
X
?0
@1
A2
&3
+4
,5
96
:7"
trackable_list_wrapper
X
?0
@1
A2
&3
+4
,5
96
:7"
trackable_list_wrapper
Ī
Bnon_trainable_variables

Clayers
regularization_losses
	variables
trainable_variables
Dmetrics
Elayer_regularization_losses
Flayer_metrics
°__call__
²_default_save_signature
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
-
Åserving_default"
signature_map


G_inbound_nodes

?kernel
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+Ę&call_and_return_all_conditional_losses
Ē__call__"ģ
_tf_keras_layerŅ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}
ō
L_inbound_nodes
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+Č&call_and_return_all_conditional_losses
É__call__"Ļ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
°
Qnon_trainable_variables

Rlayers
regularization_losses
	variables
trainable_variables
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
³__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object


V_inbound_nodes

@kernel
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
+Ź&call_and_return_all_conditional_losses
Ė__call__"ģ
_tf_keras_layerŅ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
ō
[_inbound_nodes
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+Ģ&call_and_return_all_conditional_losses
Ķ__call__"Ļ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
°
`non_trainable_variables

alayers
regularization_losses
	variables
trainable_variables
bmetrics
clayer_regularization_losses
dlayer_metrics
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object


e_inbound_nodes

Akernel
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+Ī&call_and_return_all_conditional_losses
Ļ__call__"ģ
_tf_keras_layerŅ{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
ō
j_inbound_nodes
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
+Š&call_and_return_all_conditional_losses
Ń__call__"Ļ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
°
onon_trainable_variables

players
regularization_losses
	variables
 trainable_variables
qmetrics
rlayer_regularization_losses
slayer_metrics
·__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
tnon_trainable_variables

ulayers
"regularization_losses
#	variables
$trainable_variables
vmetrics
wlayer_regularization_losses
xlayer_metrics
ŗ__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_7/kernel
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
°
ynon_trainable_variables

zlayers
'regularization_losses
(	variables
)trainable_variables
{metrics
|layer_regularization_losses
}layer_metrics
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
.:,2instance_normalization_10/gamma
-:+2instance_normalization_10/beta
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
³
~non_trainable_variables

layers
-regularization_losses
.	variables
/trainable_variables
metrics
 layer_regularization_losses
layer_metrics
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ą__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ā__call__
+Į&call_and_return_all_conditional_losses
'Į"call_and_return_conditional_losses"
_generic_user_object
": 
į2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
µ
non_trainable_variables
layers
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ä__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
6:42sequential_7/conv2d_4/kernel
6:4 2sequential_8/conv2d_5/kernel
6:4 @2sequential_9/conv2d_6/kernel
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
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
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
µ
non_trainable_variables
layers
Hregularization_losses
I	variables
Jtrainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ē__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
Mregularization_losses
N	variables
Otrainable_variables
metrics
 layer_regularization_losses
layer_metrics
É__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
µ
non_trainable_variables
layers
Wregularization_losses
X	variables
Ytrainable_variables
metrics
 layer_regularization_losses
 layer_metrics
Ė__call__
+Ź&call_and_return_all_conditional_losses
'Ź"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
”non_trainable_variables
¢layers
\regularization_losses
]	variables
^trainable_variables
£metrics
 ¤layer_regularization_losses
„layer_metrics
Ķ__call__
+Ģ&call_and_return_all_conditional_losses
'Ģ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
µ
¦non_trainable_variables
§layers
fregularization_losses
g	variables
htrainable_variables
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
Ļ__call__
+Ī&call_and_return_all_conditional_losses
'Ī"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
«non_trainable_variables
¬layers
kregularization_losses
l	variables
mtrainable_variables
­metrics
 ®layer_regularization_losses
Ælayer_metrics
Ń__call__
+Š&call_and_return_all_conditional_losses
'Š"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
Ś2×
#__inference_Dis_layer_call_fn_10476
#__inference_Dis_layer_call_fn_10294
#__inference_Dis_layer_call_fn_10497
#__inference_Dis_layer_call_fn_10244Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ę2Ć
>__inference_Dis_layer_call_and_return_conditional_losses_10193
>__inference_Dis_layer_call_and_return_conditional_losses_10386
>__inference_Dis_layer_call_and_return_conditional_losses_10455
>__inference_Dis_layer_call_and_return_conditional_losses_10164Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ķ2ź
__inference__wrapped_model_9735Ę
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *6¢3
1.
discriminator’’’’’’’’’
ü2ł
,__inference_sequential_7_layer_call_fn_10527
+__inference_sequential_7_layer_call_fn_9811
,__inference_sequential_7_layer_call_fn_10520
+__inference_sequential_7_layer_call_fn_9796Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
č2å
F__inference_sequential_7_layer_call_and_return_conditional_losses_9780
G__inference_sequential_7_layer_call_and_return_conditional_losses_10513
F__inference_sequential_7_layer_call_and_return_conditional_losses_9772
G__inference_sequential_7_layer_call_and_return_conditional_losses_10505Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ü2ł
,__inference_sequential_8_layer_call_fn_10557
+__inference_sequential_8_layer_call_fn_9872
+__inference_sequential_8_layer_call_fn_9887
,__inference_sequential_8_layer_call_fn_10550Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
č2å
F__inference_sequential_8_layer_call_and_return_conditional_losses_9856
G__inference_sequential_8_layer_call_and_return_conditional_losses_10543
F__inference_sequential_8_layer_call_and_return_conditional_losses_9848
G__inference_sequential_8_layer_call_and_return_conditional_losses_10535Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ü2ł
,__inference_sequential_9_layer_call_fn_10580
,__inference_sequential_9_layer_call_fn_10587
+__inference_sequential_9_layer_call_fn_9963
+__inference_sequential_9_layer_call_fn_9948Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
č2å
G__inference_sequential_9_layer_call_and_return_conditional_losses_10573
G__inference_sequential_9_layer_call_and_return_conditional_losses_10565
F__inference_sequential_9_layer_call_and_return_conditional_losses_9932
F__inference_sequential_9_layer_call_and_return_conditional_losses_9924Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
²2Æ
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_9970ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
/__inference_zero_padding2d_1_layer_call_fn_9976ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
ķ2ź
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10594¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_conv2d_7_layer_call_fn_10601¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
“2±
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_10020Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
9__inference_instance_normalization_10_layer_call_fn_10030Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
ņ2ļ
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_10606¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_leaky_re_lu_9_layer_call_fn_10611¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_flatten_1_layer_call_and_return_conditional_losses_10617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_flatten_1_layer_call_fn_10622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_2_layer_call_and_return_conditional_losses_10632¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_2_layer_call_fn_10641¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
8B6
#__inference_signature_wrapper_10317discriminator
ķ2ź
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10648¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_conv2d_4_layer_call_fn_10655¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ņ2ļ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_10660¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_leaky_re_lu_6_layer_call_fn_10665¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10672¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_conv2d_5_layer_call_fn_10679¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ņ2ļ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_10684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_leaky_re_lu_7_layer_call_fn_10689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10696¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_conv2d_6_layer_call_fn_10703¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ņ2ļ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_10708¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_leaky_re_lu_8_layer_call_fn_10713¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ½
>__inference_Dis_layer_call_and_return_conditional_losses_10164{?@A&+,9:H¢E
>¢;
1.
discriminator’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ½
>__inference_Dis_layer_call_and_return_conditional_losses_10193{?@A&+,9:H¢E
>¢;
1.
discriminator’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ¶
>__inference_Dis_layer_call_and_return_conditional_losses_10386t?@A&+,9:A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ¶
>__inference_Dis_layer_call_and_return_conditional_losses_10455t?@A&+,9:A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
#__inference_Dis_layer_call_fn_10244n?@A&+,9:H¢E
>¢;
1.
discriminator’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
#__inference_Dis_layer_call_fn_10294n?@A&+,9:H¢E
>¢;
1.
discriminator’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
#__inference_Dis_layer_call_fn_10476g?@A&+,9:A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
#__inference_Dis_layer_call_fn_10497g?@A&+,9:A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’¢
__inference__wrapped_model_9735?@A&+,9:@¢=
6¢3
1.
discriminator’’’’’’’’’
Ŗ "1Ŗ.
,
dense_2!
dense_2’’’’’’’’’“
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10648m?9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ "-¢*
# 
0’’’’’’’’’@@
 
(__inference_conv2d_4_layer_call_fn_10655`?9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ " ’’’’’’’’’@@²
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10672k@7¢4
-¢*
(%
inputs’’’’’’’’’@@
Ŗ "-¢*
# 
0’’’’’’’’’   
 
(__inference_conv2d_5_layer_call_fn_10679^@7¢4
-¢*
(%
inputs’’’’’’’’’@@
Ŗ " ’’’’’’’’’   ²
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10696kA7¢4
-¢*
(%
inputs’’’’’’’’’   
Ŗ "-¢*
# 
0’’’’’’’’’@
 
(__inference_conv2d_6_layer_call_fn_10703^A7¢4
-¢*
(%
inputs’’’’’’’’’   
Ŗ " ’’’’’’’’’@³
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10594l&7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ ".¢+
$!
0’’’’’’’’’
 
(__inference_conv2d_7_layer_call_fn_10601_&7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ "!’’’’’’’’’¤
B__inference_dense_2_layer_call_and_return_conditional_losses_10632^9:1¢.
'¢$
"
inputs’’’’’’’’’į
Ŗ "%¢"

0’’’’’’’’’
 |
'__inference_dense_2_layer_call_fn_10641Q9:1¢.
'¢$
"
inputs’’’’’’’’’į
Ŗ "’’’’’’’’’«
D__inference_flatten_1_layer_call_and_return_conditional_losses_10617c8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "'¢$

0’’’’’’’’’į
 
)__inference_flatten_1_layer_call_fn_10622V8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "’’’’’’’’’įė
T__inference_instance_normalization_10_layer_call_and_return_conditional_losses_10020+,J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ć
9__inference_instance_normalization_10_layer_call_fn_10030+,J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’“
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_10660h7¢4
-¢*
(%
inputs’’’’’’’’’@@
Ŗ "-¢*
# 
0’’’’’’’’’@@
 
-__inference_leaky_re_lu_6_layer_call_fn_10665[7¢4
-¢*
(%
inputs’’’’’’’’’@@
Ŗ " ’’’’’’’’’@@“
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_10684h7¢4
-¢*
(%
inputs’’’’’’’’’   
Ŗ "-¢*
# 
0’’’’’’’’’   
 
-__inference_leaky_re_lu_7_layer_call_fn_10689[7¢4
-¢*
(%
inputs’’’’’’’’’   
Ŗ " ’’’’’’’’’   “
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_10708h7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ "-¢*
# 
0’’’’’’’’’@
 
-__inference_leaky_re_lu_8_layer_call_fn_10713[7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ " ’’’’’’’’’@¶
H__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_10606j8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
-__inference_leaky_re_lu_9_layer_call_fn_10611]8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’Ą
G__inference_sequential_7_layer_call_and_return_conditional_losses_10505u?A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ "-¢*
# 
0’’’’’’’’’@@
 Ą
G__inference_sequential_7_layer_call_and_return_conditional_losses_10513u?A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’@@
 Ē
F__inference_sequential_7_layer_call_and_return_conditional_losses_9772}?I¢F
?¢<
2/
conv2d_4_input’’’’’’’’’
p

 
Ŗ "-¢*
# 
0’’’’’’’’’@@
 Ē
F__inference_sequential_7_layer_call_and_return_conditional_losses_9780}?I¢F
?¢<
2/
conv2d_4_input’’’’’’’’’
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’@@
 
,__inference_sequential_7_layer_call_fn_10520h?A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ " ’’’’’’’’’@@
,__inference_sequential_7_layer_call_fn_10527h?A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ " ’’’’’’’’’@@
+__inference_sequential_7_layer_call_fn_9796p?I¢F
?¢<
2/
conv2d_4_input’’’’’’’’’
p

 
Ŗ " ’’’’’’’’’@@
+__inference_sequential_7_layer_call_fn_9811p?I¢F
?¢<
2/
conv2d_4_input’’’’’’’’’
p 

 
Ŗ " ’’’’’’’’’@@¾
G__inference_sequential_8_layer_call_and_return_conditional_losses_10535s@?¢<
5¢2
(%
inputs’’’’’’’’’@@
p

 
Ŗ "-¢*
# 
0’’’’’’’’’   
 ¾
G__inference_sequential_8_layer_call_and_return_conditional_losses_10543s@?¢<
5¢2
(%
inputs’’’’’’’’’@@
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’   
 Å
F__inference_sequential_8_layer_call_and_return_conditional_losses_9848{@G¢D
=¢:
0-
conv2d_5_input’’’’’’’’’@@
p

 
Ŗ "-¢*
# 
0’’’’’’’’’   
 Å
F__inference_sequential_8_layer_call_and_return_conditional_losses_9856{@G¢D
=¢:
0-
conv2d_5_input’’’’’’’’’@@
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’   
 
,__inference_sequential_8_layer_call_fn_10550f@?¢<
5¢2
(%
inputs’’’’’’’’’@@
p

 
Ŗ " ’’’’’’’’’   
,__inference_sequential_8_layer_call_fn_10557f@?¢<
5¢2
(%
inputs’’’’’’’’’@@
p 

 
Ŗ " ’’’’’’’’’   
+__inference_sequential_8_layer_call_fn_9872n@G¢D
=¢:
0-
conv2d_5_input’’’’’’’’’@@
p

 
Ŗ " ’’’’’’’’’   
+__inference_sequential_8_layer_call_fn_9887n@G¢D
=¢:
0-
conv2d_5_input’’’’’’’’’@@
p 

 
Ŗ " ’’’’’’’’’   ¾
G__inference_sequential_9_layer_call_and_return_conditional_losses_10565sA?¢<
5¢2
(%
inputs’’’’’’’’’   
p

 
Ŗ "-¢*
# 
0’’’’’’’’’@
 ¾
G__inference_sequential_9_layer_call_and_return_conditional_losses_10573sA?¢<
5¢2
(%
inputs’’’’’’’’’   
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’@
 Å
F__inference_sequential_9_layer_call_and_return_conditional_losses_9924{AG¢D
=¢:
0-
conv2d_6_input’’’’’’’’’   
p

 
Ŗ "-¢*
# 
0’’’’’’’’’@
 Å
F__inference_sequential_9_layer_call_and_return_conditional_losses_9932{AG¢D
=¢:
0-
conv2d_6_input’’’’’’’’’   
p 

 
Ŗ "-¢*
# 
0’’’’’’’’’@
 
,__inference_sequential_9_layer_call_fn_10580fA?¢<
5¢2
(%
inputs’’’’’’’’’   
p

 
Ŗ " ’’’’’’’’’@
,__inference_sequential_9_layer_call_fn_10587fA?¢<
5¢2
(%
inputs’’’’’’’’’   
p 

 
Ŗ " ’’’’’’’’’@
+__inference_sequential_9_layer_call_fn_9948nAG¢D
=¢:
0-
conv2d_6_input’’’’’’’’’   
p

 
Ŗ " ’’’’’’’’’@
+__inference_sequential_9_layer_call_fn_9963nAG¢D
=¢:
0-
conv2d_6_input’’’’’’’’’   
p 

 
Ŗ " ’’’’’’’’’@ø
#__inference_signature_wrapper_10317?@A&+,9:Q¢N
¢ 
GŖD
B
discriminator1.
discriminator’’’’’’’’’"1Ŗ.
,
dense_2!
dense_2’’’’’’’’’ķ
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_9970R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
/__inference_zero_padding2d_1_layer_call_fn_9976R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’