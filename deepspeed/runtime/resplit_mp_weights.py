
import torch
import os

# Constants for weight resplitting
WORD_EMBEDDINGS_KEY = 'language_model.embedding.word_embeddings.weight'
POSITION_EMBEDDINGS_KEY = 'language_model.embedding.position_embeddings.weight'
LAYER_NORM_WEIGHT_KEY = 'language_model.transformer.layers.{}.input_layernorm.weight'
LAYER_NORM_BIAS_KEY = 'language_model.transformer.layers.{}.input_layernorm.bias'
QKV_WEIGHT_KEY = 'language_model.transformer.layers.{}.attention.query_key_value.weight'
QKV_BIAS_KEY = 'language_model.transformer.layers.{}.attention.query_key_value.bias'
POST_ATTENTION_LAYER_NORM_WEIGHT_KEY = 'language_model.transformer.layers.{}.post_attention_layernorm.weight'
POST_ATTENTION_LAYER_NORM_BIAS_KEY = 'language_model.transformer.layers.{}.post_attention_layernorm.bias'
ATTENTION_DENSE_WEIGHT_KEY = 'language_model.transformer.layers.{}.attention.dense.weight'
ATTENTION_DENSE_BIAS_KEY = 'language_model.transformer.layers.{}.attention.dense.bias'
MLP_H_2_4H_WEIGHT_KEY = 'language_model.transformer.layers.{}.mlp.dense_h_to_4h.weight'
MLP_H_2_4H_BIAS_KEY = 'language_model.transformer.layers.{}.mlp.dense_h_to_4h.bias'
MLP_4H_2_H_WEIGHT_KEY = 'language_model.transformer.layers.{}.mlp.dense_4h_to_h.weight'
MLP_4H_2_H_BIAS_KEY = 'language_model.transformer.layers.{}.mlp.dense_4h_to_h.bias'
FINAL_LAYER_NORM_WEIGHT_KEY = 'language_model.transformer.final_layernorm.weight'
FINAL_LAYER_NORM_BIAS_KEY = 'language_model.transformer.final_layernorm.bias'

# Constants for optimizer state respliting
NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS = 2
NUM_2D_TENSOR_IN_TRANSFORMER_LAYER = 4
NUM_1D_TENSOR_IN_TRANSFORMER_LAYER = 8

WORD_EMBEDDINGS_OPTIM_STATE_KEY = 0
POSITION_EMBEDDINGS_OPTIM_STATE_KEY = 1

QKV_WEIGHT_OPTIM_STATE_KEY = 2
ATTENTION_DENSE_WEIGHT_OPTIM_STATE_KEY = 3
MLP_H_2_4H_WEIGHT_OPTIM_STATE_KEY = 4
MLP_4H_2_H_WEIGHT_OPTIM_STATE_KEY = 5

INPUT_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY = 0
INPUT_LAYER_NORM_BIAS_OPTIM_STATE_KEY = 1
QKV_BIAS_OPTIM_STATE_KEY = 2
ATTENTION_DENSE_BIAS_OPTIM_STATE_KEY = 3
POST_ATTENTION_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY = 4
POST_ATTENTION_LAYER_NORM_BIAS_OPTIM_STATE_KEY = 5
MLP_H_2_4H_BIAS_OPTIM_STATE_KEY = 6
MLP_4H_2_H_BIAS_OPTIM_STATE_KEY = 7

FINAL_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY = 0
FINAL_LAYER_NORM_BIAS_OPTIM_STATE_KEY = 1

EXP_AVG_KEY = 'exp_avg'
EXP_AVG_SQ_KEY = 'exp_avg_sq'


def init_method_normal(tensor, std):
    """Init method based on N(0, sigma)."""
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def resplit_word_embeddings_weights(word_embeddings_weight_list, new_mp_size, args):
    word_embeddings_weight_all = torch.cat(word_embeddings_weight_list, dim=0)

    old_padded_vocab_size, embedding_dim = word_embeddings_weight_all.size()

    assert old_padded_vocab_size == args.padded_vocab_size, f'padded vocab size changed from {old_padded_vocab_size} to {args.padded_vocab_size}'

    per_partition_vocab_size = divide(args.padded_vocab_size, new_mp_size)

    # pad dummy rows at the end if the word embedding matrix under the new mp size has more rows now. 
    # if args.padded_vocab_size > old_padded_vocab_size:
    #     pad_size = args.padded_vocab_size - old_padded_vocab_size
    #     pad_tensor = torch.empty(pad_size, embedding_dim, device=torch.cuda.current_device(), dtype=args.params_dtype)
    #     init_method_normal(pad_tensor, args.init_method_std)
    #     word_embeddings_weight_all = torch.cat((word_embeddings_weight_all, pad_tensor), dim=0)

    new_word_embeddings_weight_list = []
    for mp_rank in range(new_mp_size):
        vocab_start_index = mp_rank * per_partition_vocab_size
        vocab_end_index = vocab_start_index + per_partition_vocab_size
        new_word_embeddings_weight_list.append(
            word_embeddings_weight_all[vocab_start_index : vocab_end_index, :]
        )
    return new_word_embeddings_weight_list


def resplit_column_parallel_linear_weights(cp_weight_list, new_mp_size):
    cp_weight_all = torch.cat(cp_weight_list, dim=0)

    global_output_size, _ = cp_weight_all.size()
    output_size_per_partition = divide(global_output_size, new_mp_size)

    new_cp_weight_list = []
    for mp_rank in range(new_mp_size):
        start_index = mp_rank * output_size_per_partition
        end_index = start_index + output_size_per_partition
        new_cp_weight_list.append(
            cp_weight_all[start_index : end_index, :]
        )
    return new_cp_weight_list


def resplit_column_parallel_linear_biases(cp_bias_list, new_mp_size):
    cp_bias_all = torch.cat(cp_bias_list)
    
    global_bias_size = cp_bias_all.size()[0]
    bias_size_per_partition = divide(global_bias_size, new_mp_size)

    new_cp_bias_list = []
    for mp_rank in range(new_mp_size):
        start_index = mp_rank * bias_size_per_partition
        end_index = start_index + bias_size_per_partition
        new_cp_bias_list.append(
            cp_bias_all[start_index : end_index]
        )
    return new_cp_bias_list


def resplit_row_parallel_linear_weights(rp_weight_list, new_mp_size):
    rp_weight_all = torch.cat(rp_weight_list, dim=1)

    _, global_input_size = rp_weight_all.size()
    input_size_per_partition = divide(global_input_size, new_mp_size)

    new_rp_weight_list = []
    for mp_rank in range(new_mp_size):
        start_index = mp_rank * input_size_per_partition
        end_index = start_index + input_size_per_partition
        new_rp_weight_list.append(
            rp_weight_all[:, start_index : end_index]
        )
    return new_rp_weight_list


def parse_num_trans_layers(state_dict):
    '''
    hacky string-parsing solution to getting the number of transformer layers
    might have better a solution
    '''
    trans_layer_keys = [key for key in state_dict.keys() if 'language_model.transformer.layers' in key]
    layer_numbers = [int(key.split('.')[3]) for key in trans_layer_keys]
    num_trans_layers = sorted(layer_numbers)[-1] + 1
    return num_trans_layers


def resplit_state_dicts(state_dict_list, new_mp_size, args):
    new_state_dict_list = [{} for _ in range(new_mp_size)]
    arbitrary_state_dict = state_dict_list[0]
    # print(f'arbitrary_state_dict = {arbitrary_state_dict}')

    # resplit the word embeddings weights
    word_embeddings_weight_list = [state_dict[WORD_EMBEDDINGS_KEY] for state_dict in state_dict_list]
    new_word_embeddings_weight_list = resplit_word_embeddings_weights(
        word_embeddings_weight_list=word_embeddings_weight_list, 
        new_mp_size=new_mp_size, 
        args=args
    )
    for state_dict, new_word_embeddings_weight in zip(new_state_dict_list, new_word_embeddings_weight_list):
        state_dict[WORD_EMBEDDINGS_KEY] = new_word_embeddings_weight
    
    # Since there is no model parallelism for position embeddings,
    # we do not need to resplit the position embeddings weights. 
    # We simply reload the position embeddings weights. 
    position_embeddings_weight = arbitrary_state_dict[POSITION_EMBEDDINGS_KEY]
    for state_dict in new_state_dict_list:
        state_dict[POSITION_EMBEDDINGS_KEY] = position_embeddings_weight.clone()

    # loop over all transformer layers
    num_trans_layers = parse_num_trans_layers(state_dict_list[0])
    assert num_trans_layers == args.num_layers, f'[*] number of layers does not equal to number of layers in checkpoint: {num_trans_layers} != {args.num_layers}'
    for layer_num in range(num_trans_layers):
        # Since there is no model parallelism for ParallelTransformerLayer.input_layernorm, 
        # we do not need to resplit ParallelTransformerLayer.input_layernorm's weights and biases. 
        # We simply reload ParallelTransformerLayer.input_layernorm's weights and biases
        layer_norm_weight_key = LAYER_NORM_WEIGHT_KEY.format(layer_num)
        layer_norm_bias_key = LAYER_NORM_BIAS_KEY.format(layer_num)
        layer_norm_weight = arbitrary_state_dict[layer_norm_weight_key]
        layer_norm_bias = arbitrary_state_dict[layer_norm_bias_key]
        for state_dict in new_state_dict_list:
            state_dict[layer_norm_weight_key] = layer_norm_weight.clone()
            state_dict[layer_norm_bias_key] = layer_norm_bias.clone()
        
        # resplit the self-attention QKV weights
        qkv_weight_key = QKV_WEIGHT_KEY.format(layer_num)
        old_qkv_weight_list = [old_state_dict[qkv_weight_key] for old_state_dict in state_dict_list]
        new_qkv_weight_list = resplit_column_parallel_linear_weights(
            cp_weight_list=old_qkv_weight_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_qkv_weight in zip(new_state_dict_list, new_qkv_weight_list):
            state_dict[qkv_weight_key] = new_qkv_weight

        # resplit the self-attention QKV biases
        qkv_bias_key = QKV_BIAS_KEY.format(layer_num)
        old_qkv_bias_list = [old_state_dict[qkv_bias_key] for old_state_dict in state_dict_list]
        new_qkv_bias_list = resplit_column_parallel_linear_biases(
            cp_bias_list=old_qkv_bias_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_qkv_bias in zip(new_state_dict_list, new_qkv_bias_list):
            state_dict[qkv_bias_key] = new_qkv_bias
        
        # resplit the self-attention dense weights
        attention_dense_weight_key = ATTENTION_DENSE_WEIGHT_KEY.format(layer_num)
        old_attention_dense_weight_list = [old_state_dict[attention_dense_weight_key] for old_state_dict in state_dict_list]
        new_attention_dense_weight_list = resplit_row_parallel_linear_weights(
            rp_weight_list=old_attention_dense_weight_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_attention_dense_weight in zip(new_state_dict_list, new_attention_dense_weight_list):
            state_dict[attention_dense_weight_key] = new_attention_dense_weight

        # Since there is no model parallelism for ParallelSelfAttention dense biases, 
        # we do not need to resplit the ParallelSelfAttention.dense biases.
        # We simply reload the ParallelSelfAttention.dense biases.
        attention_dense_bias_key = ATTENTION_DENSE_BIAS_KEY.format(layer_num)
        attention_dense_bias = arbitrary_state_dict[attention_dense_bias_key]
        for state_dict in new_state_dict_list:
            state_dict[attention_dense_bias_key] = attention_dense_bias.clone()

        # Since there is no model parallelism for ParallelTransformerLayer.post_attention_layernorm, 
        # we do not need to resplit ParallelTransformerLayer.post_attention_layernorm's weights and biases. 
        # We simply reload ParallelTransformerLayer.post_attention_layernorm's weights and biases
        post_attention_layer_norm_weight_key = POST_ATTENTION_LAYER_NORM_WEIGHT_KEY.format(layer_num)
        post_attention_layer_norm_bias_key = POST_ATTENTION_LAYER_NORM_BIAS_KEY.format(layer_num)
        post_attention_layer_norm_weight = arbitrary_state_dict[post_attention_layer_norm_weight_key]
        post_attention_layer_norm_bias = arbitrary_state_dict[post_attention_layer_norm_bias_key]
        for state_dict in new_state_dict_list:
            state_dict[post_attention_layer_norm_weight_key] = post_attention_layer_norm_weight.clone()
            state_dict[post_attention_layer_norm_bias_key] = post_attention_layer_norm_bias.clone()
        
        # resplit the mlp dense_h_to_4h weights
        mlp_h_2_4h_weight_key = MLP_H_2_4H_WEIGHT_KEY.format(layer_num)
        old_mlp_h_2_4h_weight_list = [old_state_dict[mlp_h_2_4h_weight_key] for old_state_dict in state_dict_list]
        new_mlp_h_2_4h_weight_list = resplit_column_parallel_linear_weights(
            cp_weight_list=old_mlp_h_2_4h_weight_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_mlp_h_2_4h_weight in zip(new_state_dict_list, new_mlp_h_2_4h_weight_list):
            state_dict[mlp_h_2_4h_weight_key] = new_mlp_h_2_4h_weight

        # resplit the mlp dense_h_to_4h biases
        mlp_h_2_4h_bias_key = MLP_H_2_4H_BIAS_KEY.format(layer_num)
        old_mlp_h_2_4h_bias_list = [old_state_dict[mlp_h_2_4h_bias_key] for old_state_dict in state_dict_list]
        new_mlp_h_2_4h_bias_list = resplit_column_parallel_linear_biases(
            cp_bias_list=old_mlp_h_2_4h_bias_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_mlp_h_2_4h_bias in zip(new_state_dict_list, new_mlp_h_2_4h_bias_list):
            state_dict[mlp_h_2_4h_bias_key] = new_mlp_h_2_4h_bias

        # resplit the mlp dense_4h_to_h weights
        mlp_4h_2_h_weight_key = MLP_4H_2_H_WEIGHT_KEY.format(layer_num)
        old_mlp_4h_2_h_weight_list = [old_state_dict[mlp_4h_2_h_weight_key] for old_state_dict in state_dict_list]
        new_mlp_4h_2_h_weight_list = resplit_row_parallel_linear_weights(
            rp_weight_list=old_mlp_4h_2_h_weight_list, 
            new_mp_size=new_mp_size
        )
        for state_dict, new_mlp_4h_2_h_weight in zip(new_state_dict_list, new_mlp_4h_2_h_weight_list):
            state_dict[mlp_4h_2_h_weight_key] = new_mlp_4h_2_h_weight

        # Since there is no model parallelism for ParallelMLP.dense_4h_to_h, 
        # we do not neet to resplit the ParallelMLP.dense_4h_to_h biases.
        # We simply reload the ParallelMLP.dense_4h_to_h biases
        mlp_4h_2_h_bias_key = MLP_4H_2_H_BIAS_KEY.format(layer_num)
        mlp_4h_2_h_bias = arbitrary_state_dict[mlp_4h_2_h_bias_key]
        for state_dict in new_state_dict_list:
            state_dict[mlp_4h_2_h_bias_key] = mlp_4h_2_h_bias.clone()
    
    final_layer_norm_weight = arbitrary_state_dict[FINAL_LAYER_NORM_WEIGHT_KEY]
    final_layer_norm_bias = arbitrary_state_dict[FINAL_LAYER_NORM_BIAS_KEY]
    for state_dict in new_state_dict_list:
        state_dict[FINAL_LAYER_NORM_WEIGHT_KEY] = final_layer_norm_weight.clone()
        state_dict[FINAL_LAYER_NORM_BIAS_KEY] = final_layer_norm_bias.clone()

    return new_state_dict_list


def resplit_optim_states(old_optim_state_list, new_mp_size, args):
    # resplit the word embedding optim states
    new_optim_state_list = [{} for _ in range(new_mp_size)]

    # resplit the word embeeddings optim state: exp avg
    old_word_embeddings_exp_avg_list = [old_optim_state[WORD_EMBEDDINGS_OPTIM_STATE_KEY][EXP_AVG_KEY] for old_optim_state in old_optim_state_list]
    new_word_embeddings_exp_avg_list = resplit_word_embeddings_weights(
        word_embeddings_weight_list=old_word_embeddings_exp_avg_list, 
        new_mp_size=new_mp_size, 
        args=args
    )

    # resplit the word embeddings optim state: exp avg sq
    old_word_embeddings_exp_avg_sq_list = [old_optim_state[WORD_EMBEDDINGS_OPTIM_STATE_KEY][EXP_AVG_SQ_KEY] for old_optim_state in old_optim_state_list]
    new_word_embeddings_exp_avg_sq_list = resplit_word_embeddings_weights(
        word_embeddings_weight_list=old_word_embeddings_exp_avg_sq_list, 
        new_mp_size=new_mp_size, 
        args=args
    )
    for new_optim_state, new_word_embeddings_exp_avg, new_word_embeddings_exp_avg_sq in zip(
        new_optim_state_list, new_word_embeddings_exp_avg_list, new_word_embeddings_exp_avg_sq_list
    ):
        new_optim_state[WORD_EMBEDDINGS_OPTIM_STATE_KEY] = {
            EXP_AVG_KEY: new_word_embeddings_exp_avg, 
            EXP_AVG_SQ_KEY: new_word_embeddings_exp_avg_sq
        }
    
    # Since there is no model parallelism for position embeddings, 
    # we do not need to resplit the position embedding exp_avg and exp_avg_sq
    # we simply reload the position embeddings exp_avg and exp_avg_sq
    reload_optim_states_for_key(
        key=POSITION_EMBEDDINGS_OPTIM_STATE_KEY, 
        old_optim_state_list=old_optim_state_list, 
        new_optim_state_list=new_optim_state_list
    )
    
    # loop over all transformer layers
    for layer_num in range(args.num_layers):
        # Since there is no model parallelism for ParallelTransformerLayer.input_layernorm, 
        # we do not need to resplit ParallelTransformerLayer.input_layernorm's exp_avg and exp_avg_sq. 
        # We simply reload ParallelTransformerLayer.input_layernorm's exp_avg and exp_avg_sq
        input_layer_norm_weight_optim_state_key = INPUT_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY \
                                                  + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                                  + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                                  + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=input_layer_norm_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )
        
        input_layer_norm_bias_optim_state_key = INPUT_LAYER_NORM_BIAS_OPTIM_STATE_KEY \
                                                + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                                + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                                + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=input_layer_norm_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )

        # resplit the self-attention QKV weights optim states
        qkv_weight_optim_state_key = QKV_WEIGHT_OPTIM_STATE_KEY + layer_num * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=qkv_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_column_parallel_linear_weights
        )
        
        # resplit the self-attention QKV biases optim states
        qkv_bias_optim_state_key = QKV_BIAS_OPTIM_STATE_KEY \
                       + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                       + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                       + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=qkv_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_column_parallel_linear_biases
        )
        
        # resplit the self-attention dense weights optim states
        attention_dense_weight_optim_state_key = ATTENTION_DENSE_WEIGHT_OPTIM_STATE_KEY + layer_num * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=attention_dense_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_row_parallel_linear_weights
        )
        
        # Since there is no model parallelism for ParallelSelfAttention dense biases, 
        # we do not need to resplit the ParallelSelfAttention.dense biases optim states.
        # We simply reload the ParallelSelfAttention.dense biases optim states.
        attention_dense_bias_optim_state_key = ATTENTION_DENSE_BIAS_OPTIM_STATE_KEY + \
                                               + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                               + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                               + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=attention_dense_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )
        
        # Since there is no model parallelism for ParallelTransformerLayer.post_attention_layernorm, 
        # we do not need to resplit ParallelTransformerLayer.post_attention_layernorm's weights and biases optim states. 
        # We simply reload ParallelTransformerLayer.post_attention_layernorm's weights and biases optim states.
        post_attention_layer_norm_weight_optim_state_key = POST_ATTENTION_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY + \
                                                           + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                                           + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                                           + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=post_attention_layer_norm_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )
        
        post_attention_layer_norm_bias_optim_state_key = POST_ATTENTION_LAYER_NORM_BIAS_OPTIM_STATE_KEY + \
                                                         + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                                         + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                                         + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=post_attention_layer_norm_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )
        
        # resplit the mlp dense_h_to_4h weights optim states
        mlp_h_2_4h_weight_optim_state_key = MLP_H_2_4H_WEIGHT_OPTIM_STATE_KEY + layer_num * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=mlp_h_2_4h_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_column_parallel_linear_weights
        )
        
        # resplit the mlp dense_h_to_4h biases optim states
        mlp_h_to_4h_bias_optim_state_key = MLP_H_2_4H_BIAS_OPTIM_STATE_KEY + \
                                           + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                           + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                           + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=mlp_h_to_4h_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_column_parallel_linear_biases
        )

        # resplit the mlp dense_4h_to_h weights optim states
        mlp_4h_to_h_weight_optim_state_key = MLP_4H_2_H_WEIGHT_OPTIM_STATE_KEY + layer_num * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER
        resplit_optim_states_for_key(
            key=mlp_4h_to_h_weight_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list, 
            new_mp_size=new_mp_size, 
            resplit_func=resplit_row_parallel_linear_weights
        )

        # Since there is no model parallelism for ParallelMLP.dense_4h_to_h, 
        # we do not neet to resplit the ParallelMLP.dense_4h_to_h biases optim states.
        # We simply reload the ParallelMLP.dense_4h_to_h biases optim states.
        mlp_4h_to_h_bias_optim_state_key = MLP_4H_2_H_BIAS_OPTIM_STATE_KEY + \
                                           + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS \
                                           + args.num_layers * NUM_2D_TENSOR_IN_TRANSFORMER_LAYER \
                                           + layer_num * NUM_1D_TENSOR_IN_TRANSFORMER_LAYER
        reload_optim_states_for_key(
            key=mlp_4h_to_h_bias_optim_state_key, 
            old_optim_state_list=old_optim_state_list, 
            new_optim_state_list=new_optim_state_list
        )
    
    final_layer_norm_weight_optim_state_key = FINAL_LAYER_NORM_WEIGHT_OPTIM_STATE_KEY + \
                                              + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS + \
                                              + args.num_layers * (NUM_2D_TENSOR_IN_TRANSFORMER_LAYER + NUM_1D_TENSOR_IN_TRANSFORMER_LAYER)
    reload_optim_states_for_key(
        key=final_layer_norm_weight_optim_state_key, 
        old_optim_state_list=old_optim_state_list, 
        new_optim_state_list=new_optim_state_list
    )
    final_layer_norm_bias_optim_state_key = FINAL_LAYER_NORM_BIAS_OPTIM_STATE_KEY + \
                                            + NUM_TENSOR_BEFORE_TRANSFORMER_LAYERS + \
                                            + args.num_layers * (NUM_2D_TENSOR_IN_TRANSFORMER_LAYER + NUM_1D_TENSOR_IN_TRANSFORMER_LAYER)
    reload_optim_states_for_key(
        key=final_layer_norm_bias_optim_state_key, 
        old_optim_state_list=old_optim_state_list, 
        new_optim_state_list=new_optim_state_list
    )

    return new_optim_state_list


def resplit_optim_states_for_key(key, old_optim_state_list, new_optim_state_list, new_mp_size, resplit_func):
    old_exp_avg_list = [old_optim_state[key][EXP_AVG_KEY] for old_optim_state in old_optim_state_list]
    new_exp_avg_list = resplit_func(
        old_exp_avg_list, 
        new_mp_size
    )
    old_exp_avg_sq_list = [old_optim_state[key][EXP_AVG_SQ_KEY] for old_optim_state in old_optim_state_list]
    new_exp_avg_sq_list = resplit_func(
        old_exp_avg_sq_list, 
        new_mp_size
    )
    for new_optim_state, new_exp_avg, new_exp_avg_sq in zip(
        new_optim_state_list, new_exp_avg_list, new_exp_avg_sq_list
    ):
        new_optim_state[key] = {
            EXP_AVG_KEY: new_exp_avg, 
            EXP_AVG_SQ_KEY: new_exp_avg_sq
        }

def reload_optim_states_for_key(key, old_optim_state_list, new_optim_state_list):
    arbitrary_optim_state = old_optim_state_list[0]
    exp_avg = arbitrary_optim_state[key][EXP_AVG_KEY]
    exp_avg_sq = arbitrary_optim_state[key][EXP_AVG_SQ_KEY]
    for new_optim_state in new_optim_state_list:
        new_optim_state[key] = {
            EXP_AVG_KEY: exp_avg.clone(), 
            EXP_AVG_SQ_KEY: exp_avg_sq.clone()
        }


def resplit_checkpoints(checkpoint_list, new_mp_size, args):
    assert checkpoint_list, '[*] checkpoint_list cannot be empty'

    # resplit the state dicts
    state_dict_list = [checkpoint['module'] for checkpoint in checkpoint_list]
    new_state_dict_list = resplit_state_dicts(state_dict_list=state_dict_list, new_mp_size=new_mp_size, args=args)

    # resplit the optimizer states
    old_optim_state_list = [checkpoint['optimizer']['state'] for checkpoint in checkpoint_list]
    new_optim_state_list = resplit_optim_states(old_optim_state_list=old_optim_state_list, new_mp_size=new_mp_size, args=args)

    new_checkpoint_list = [checkpoint_list[mp_rank] for mp_rank in range(new_mp_size)]

    for checkpoint, state_dict, optim_state in zip(new_checkpoint_list, new_state_dict_list, new_optim_state_list):
        checkpoint['module'] = state_dict
        checkpoint['optimizer']['state'] = optim_state
    
    return new_checkpoint_list


def checkpoints_respliter(load_dir, tag, new_mp_size, args):
    checkpoint_list = []
    ckpt_dir = os.path.join(load_dir, str(tag))
    for checkpoint_filename in sorted(os.listdir(ckpt_dir)):
        if 'model_states.pt' in checkpoint_filename:
            checkpoint_list.append(
                torch.load(os.path.join(load_dir, str(tag), checkpoint_filename), map_location=lambda storage, loc: storage)
            )
    
    new_checkpoint_list = resplit_checkpoints(checkpoint_list, new_mp_size, args)
    return new_checkpoint_list
