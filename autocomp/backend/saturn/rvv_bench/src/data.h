#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned char input[];
extern unsigned int input_size;

extern unsigned char conv_weight[];
extern unsigned int conv_weight_size;

extern unsigned char conv_bias[];
extern unsigned int conv_bias_size;

extern unsigned char extra_bias[];
extern unsigned int extra_bias_size;

extern unsigned char expected_output[];
extern unsigned int expected_output_size;

#ifdef __cplusplus
}
#endif
