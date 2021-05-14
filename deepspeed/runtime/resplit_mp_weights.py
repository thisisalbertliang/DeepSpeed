import torch
import os

WORD_EMBEDDINGS_KEY = 'language_model.embedding.word_embeddings.weight'
POSITION_EMBEDDINGS_KEY = 'language_model.embedding.position_embeddings.weight'
LAYER_NORM_WEIGHT_KEY = 'language_model.transformer.layers.{}.input_layernorm.weight'
LAYER_NORM_BIAS_KEY = 'language_model.transformer.layers.{}.input_layernorm.bias'
QKV_WEIGHT_KEY = 'language_model.transformer.layers.{}.attention.query_key_value.weight'
QKV_BIAS_KEY = 'language_model.transformer.layers.{}.attention.query_key_value.bias'
POST_ATTENTION_LAYER_NORM_WEIGHT_KEY = 'language_model.transformer.layers.{}.post_attention_layernorm.weight'
POST_ATTENTION_LAYER_NORM_BIAS_KEY = 'language_model.transformer.layers.{}.post_attention_layernorm.bias'

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def resplit_word_embeddings_weights(word_embeddings_weight_list, new_mp_size):
    word_embeddings_weight_all = torch.cat(word_embeddings_weight_list, dim=0)

    global_vocab_size, _ = word_embeddings_weight_all.size()
    per_partition_vocab_size = divide(global_vocab_size, new_mp_size)

    new_word_embeddings_weight_list = []
    for mp_rank in range(new_mp_size):
        vocab_start_index = mp_rank * per_partition_vocab_size
        vocab_end_index = vocab_start_index + per_partition_vocab_size
        new_word_embeddings_weight_list.append(
            word_embeddings_weight_all[vocab_start_index : vocab_end_index, :]
        )
    return new_word_embeddings_weight_list


def resplit_self_attention_QKV_weights(qkv_weight_list, new_mp_size):
    qkv_weight_all = torch.cat(qkv_weight_list, dim=0)

    global_output_size, _ = qkv_weight_all.size()
    output_size_per_partition = divide(global_output_size, new_mp_size)

    new_qkv_weight_list = []
    for mp_rank in range(new_mp_size):
        start_index = mp_rank * output_size_per_partition
        end_index = start_index + output_size_per_partition
        new_qkv_weight_list.append(
            qkv_weight_all[start_index : end_index, :]
        )
    return new_qkv_weight_list


def resplit_self_attention_QKV_biases(qkv_bias_list, new_mp_size):
    qkv_bias_all = torch.cat(qkv_bias_list)
    
    print(f'qkv_bias_all.size() = {qkv_bias_all.size()}')
    global_bias_size = qkv_bias_all.size()[0]
    bias_size_per_partition = divide(global_bias_size, new_mp_size)

    new_qkv_bias_list = []
    for mp_rank in range(new_mp_size):
        start_index = mp_rank * bias_size_per_partition
        end_index = start_index + bias_size_per_partition
        new_qkv_bias_list.append(
            qkv_bias_all[start_index : end_index]
        )
    return new_qkv_bias_list


def resplit_self_attention_dense_weights(dense_weight_list, new_mp_size):
    # TO DO
    pass


def resplit_self_attention_dense_biases(dense_bias_list, new_mp_size):
    # TO DO
    pass


def resplit_dense_h_to_4h_weights(dense_weight_list, new_mp_size):
    # TO DO
    pass


def resplit_dense_h_to_4h_biases(dense_bias_list, new_mp_size):
    # TO DO
    pass


def resplit_dense_4h_to_h_weights(dense_weight_list, new_mp_size):
    # TO DO
    pass


def resplit_dense_4h_to_h_biases(dense_bias_list, new_mp_size):
    # TO DO
    pass


'''
More resplitting functions here later...
'''


def parse_num_trans_layers(state_dict):
    '''
    hacky string-parsing solution to getting the number of transformer layers
    might have better a solution
    '''
    trans_layer_keys = [key for key in state_dict.keys() if 'language_model.transformer.layers' in key]
    layer_numbers = [int(key.split('.')[3]) for key in trans_layer_keys]
    num_trans_layers = sorted(layer_numbers)[-1]
    return num_trans_layers


def resplit_state_dicts(state_dict_list, new_mp_size):
    new_state_dict_list = [{} for _ in range(new_mp_size)]
    arbitrary_state_dict = state_dict_list[0]
    print(f'arbitrary_state_dict = {arbitrary_state_dict}')

    # resplit the word embeddings weights
    word_embeddings_weight_list = [state_dict[WORD_EMBEDDINGS_KEY] for state_dict in state_dict_list]
    new_word_embeddings_weight_list = resplit_word_embeddings_weights(word_embeddings_weight_list=word_embeddings_weight_list, new_mp_size=new_mp_size)
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
    for layer_num in range(num_trans_layers):
        # Since there is no model parallelism for transformer layer's layer normalization, 
        # we do not need to resplit the layer normalization weights and biases. 
        # We simply reload the layer normalization weights and biases
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
        new_qkv_weight_list = resplit_self_attention_QKV_weights(qkv_weight_list=old_qkv_weight_list, new_mp_size=new_mp_size)
        for state_dict, new_qkv_weight in zip(new_state_dict_list, new_qkv_weight_list):
            state_dict[qkv_weight_key] = new_qkv_weight

        # resplit the self-attention QKV biases
        qkv_bias_key = QKV_BIAS_KEY.format(layer_num)
        old_qkv_bias_list = [old_state_dict[qkv_bias_key] for old_state_dict in state_dict_list]
        new_qkv_bias_list = resplit_self_attention_QKV_biases(qkv_bias_list=old_qkv_bias_list, new_mp_size=new_mp_size)
        for state_dict, new_qkv_bias in zip(new_state_dict_list, new_qkv_bias_list):
            state_dict[qkv_bias_key] = new_qkv_bias
        
        # resplit the self-attention dense weights
        # to do ...

        # resplit the self-attention dense biases
        # to do ...

        # Since there is no model parallelism for transformer layer's post attention layer normalization, 
        # we do not need to resplit the post attention layer normalization weights and biases. 
        # We simply reload the post attention layer normalization weights and biases
        post_attention_layer_norm_weight_key = POST_ATTENTION_LAYER_NORM_WEIGHT_KEY.format(layer_num)
        post_attention_layer_norm_bias_key = POST_ATTENTION_LAYER_NORM_BIAS_KEY.format(layer_num)
        post_attention_layer_norm_weight = arbitrary_state_dict[post_attention_layer_norm_weight_key]
        post_attention_layer_norm_bias = arbitrary_state_dict[post_attention_layer_norm_bias_key]
        for state_dict in new_state_dict_list:
            state_dict[post_attention_layer_norm_weight_key] = post_attention_layer_norm_weight.clone()
            state_dict[post_attention_layer_norm_bias_key] = post_attention_layer_norm_bias.clone()
        
        # resplit the mlp dense_h_to_4h weights
        # to do ...

        # resplit the mlp dense_h_to_4h biases
        # to do ...

        # resplit the mlp dense_4h_to_h weights
        # to do ...

        # resplit the mlp dense_4h_to_h biases
        # to do ...

    # print(f'new_state_dict_list = {new_state_dict_list}')

    # temporary garbage code to avoid errors in testing
    for key, val in state_dict_list[0].items():
        for state_dict in new_state_dict_list:
            if key not in state_dict:
                state_dict[key] = val

    return new_state_dict_list


def resplit_checkpoints(checkpoint_list, new_mp_size):
    assert checkpoint_list, 'checkpoint_list cannot be empty'

    state_dict_list = [checkpoint['module'] for checkpoint in checkpoint_list]
    new_state_dict_list = resplit_state_dicts(state_dict_list=state_dict_list, new_mp_size=new_mp_size)

    new_checkpoint_list = [checkpoint_list[mp_rank] for mp_rank in range(new_mp_size)]
    for checkpoint, state_dict in zip(new_checkpoint_list, new_state_dict_list):
        checkpoint['module'] = state_dict
    
    return new_checkpoint_list


def checkpoints_respliter(load_dir, tag, new_mp_size):
    checkpoint_list = []
    ckpt_dir = os.path.join(load_dir, str(tag))
    for checkpoint_filename in sorted(os.listdir(ckpt_dir)):
        if 'model_states.pt' in checkpoint_filename:
            checkpoint_list.append(
                torch.load(os.path.join(load_dir, str(tag), checkpoint_filename), map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            )
    # for old_mp_rank in range(old_mp_size):
    #     checkpoint_filename = os.path.join(load_dir, str(tag), f'mp_rank_0{old_mp_rank}_model_states.pt')
    #     checkpoint_list.append(
    #         torch.load(checkpoint_filename, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
    #     )
    print(f'ALBERT_DEBUG: RANK {torch.distributed.get_rank()}: checkpoint_list[0]["module] = {checkpoint_list[0]["module"]}')
    print(f'ALBERT_DEBUG: RANK {torch.distributed.get_rank()}: checkpoint_list[1]["module] = {checkpoint_list[1]["module"]}')
    
    new_checkpoint_list = resplit_checkpoints(checkpoint_list, new_mp_size)
    return new_checkpoint_list

    # for mp_rank, checkpoint in enumerate(new_checkpoint_list):
    #     checkpoint_filename = os.path.join(load_dir, str(tag), f'mp_rank_0{mp_rank}_model_states.pt')
    #     os.remove(checkpoint_filename)
    #     torch.save(checkpoint, checkpoint_filename)



# if __name__ == '__main__':
#     # checkpoint_0 = torch.load('./100/mp_rank_00_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_1 = torch.load('./100/mp_rank_01_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_2 = torch.load('./100/mp_rank_02_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_3 = torch.load('./100/mp_rank_03_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_4 = torch.load('./100/mp_rank_04_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_5 = torch.load('./100/mp_rank_05_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_6 = torch.load('./100/mp_rank_06_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))
#     # checkpoint_7 = torch.load('./100/mp_rank_07_model_states.pt', map_location=lambda storage, loc: storage.cuda(0))

#     checkpoint_list = [torch.load(f'./100/mp_rank_0{mp_rank}_model_states.pt', map_location=lambda storage, loc: storage.cuda(0)) for mp_rank in range(8)]
#     # state_dict_list = [checkpoint['module'] for checkpoint in checkpoint_list]
#     new_checkpoint_list = resplit_checkpoints(checkpoint_list=checkpoint_list, new_mp_size=4)
#     # for mp_rank, checkpoint in enumerate(new_checkpoint_list):
#     #     # torch.save(checkpoint, f'mp_rank_0{mp_rank}_model_states.pt')
#     #     os.remove(f'mp_rank_0{mp_rank}_model_states.pt')
#     # for checkpoint in new_checkpoint_list:
#     #     state_dict = checkpoint['module']
#     #     for key, val in state_dict.items():
#     #         print(f'key = {key} ; type(val) = {type(val)} ; val.size() = {val.size()}')


#     # for key, val in checkpoint_0.items():
#     #     print(f'key = {key} ; type(val) = {type(val)}')

#     # state_dict_0 = checkpoint_0['module']
#     # for key, val in state_dict_0.items():
#     #     print(f'key = {key} ; type(val) = {type(val)} ; val.size() = {val.size()}')
#     # for state_dict1, state_dict2 in zip(state_dict_list, state_dict_list[1:]):
#     #     assert torch.equal(state_dict1[POSITION_EMBEDDINGS_KEY], state_dict2[POSITION_EMBEDDINGS_KEY])