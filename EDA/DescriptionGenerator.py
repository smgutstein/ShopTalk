import argparse
import configparser
import gzip
import json
import shutil

from collections import defaultdict
from pathlib import Path
from preprocessor import Preprocessor


class DescriptionGenerator:
    def __init__(self, root_dir, listing_dir, data_file, 
                 country, language, config_file):

        self.data_path = Path(root_dir, listing_dir, data_file)
        self.country = country
        self.language = language
        self.blurb_dict = dict()

        if not Path(config_file).exists():
            print(f"Config file {config_file} not found. Assuming full text generation")
            self.config = None
        else:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
            print(f"Read config file {config_file}")

    def filter_for_country(self):
        '''Init product list filtered for items available in
           chosen country. Returns a list of dicts'''
        country_list = []
        with open(self.data_path, "r", encoding="utf-8") as file:
            for ctr,line in enumerate(file):
                new_item = json.loads(line)
                if new_item['country'] == self.country:
                    country_list.append(new_item)
        print(f"Found {len(country_list)} entries")
        return country_list

    def filter_for_language(self, in_list):
        '''Filter for language in product list. Generally each product is
           characterized by large dict. If value associated with key is list
           of dicts, then want to select dict with correct value associated 
           with language_tag key.'''
        out_list = []
        for curr_dict in in_list:
            filt_dict = dict()
            for curr_key in sorted(curr_dict.keys()):
                if isinstance(curr_dict[curr_key], list):
                    new_list = []
                    for xx in curr_dict[curr_key]:
                        if not isinstance(xx, dict):
                            new_list.append(xx)
                        elif 'language_tag' not in xx.keys():
                            new_list.append(xx)
                        elif xx['language_tag'] == self.language:
                            new_list.append(xx)
                    filt_dict[curr_key] = new_list
                else:
                    filt_dict[curr_key] = curr_dict[curr_key]
            out_list.append(filt_dict)
        return out_list
    
    def get_filtered_product_list(self):
        '''Get filtered product list for country and language'''
        country_list = self.filter_for_country()
        country_lang_list = self.filter_for_language(country_list)

        return country_lang_list
    
    def make_data_dict(self):
        ''' Convert list of dicts into single dict that uses
            "item_id" as key that maps to rest of dict'''

        self.item_id_dict = dict()

        product_list = self.get_filtered_product_list()
        
        # Loop thru item dicts in product_list
        for curr_item in product_list:
            curr_id = curr_item['item_id']
            info_dict = dict()
            
            for curr_key in curr_item.keys():
                if curr_key == 'item_id':
                    continue
                info_dict[curr_key] = curr_item[curr_key]
            self.item_id_dict[curr_id] = info_dict

            # Init field with text for LLM
            self.item_id_dict[curr_id]['llm_str'] = ""
            self.item_id_dict[curr_id]['feature_fields'] = dict()
        return 
    
    def get_all_text(self): 
        '''Get text for each item'''

        # Create initial sentence naming each item, its ID and
        # if applicable - its brand
        self.get_item_name_and_brand_text()

        # Add sentence for each desired item property
        if self.config is None:
            print("No config file found. Generating all text options")
            self.get_product_type_text()
            self.get_product_desc_text()
            self.get_weight_text()
            self.get_dimensions_text()
            self.get_color_text()
            self.get_material_text()
            self.get_node_text()
            self.get_fabric_text()
            self.get_model_year_text()
            self.get_finish_text()
            self.get_pattern_text()
            self.get_shape_text()
            self.get_bullet_point_text()  
        else:
            for text_option in self.config['Text Options']:
                if self.config.getboolean('Text Options', text_option, fallback=False):
                    method_name = f"get_{text_option}"
                    method = getattr(self, method_name, None)
                    if method is None:
                        raise ValueError(
                            f"Unknown text generation option in config: {text_option}"
                        )
                    print(f"  Generating {text_option}")
                    method()

            
        return  

        
    
    def get_item_name_and_brand_text(self):
        # Clean data for item name & construct first 2 sentences of 
        # Reference text.
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ''
            if 'item_name' in self.item_id_dict[curr_id].keys():
                prod_str = self.item_id_dict[curr_id]['item_name'][0]['value']
                prod_str = prod_str.replace('Amazon Essentials','')
                prod_str = prod_str.replace('AmazonBasics -','') 
                prod_str = prod_str.replace('AmazonBasics','')
                prod_str = prod_str.replace('Amazon Brand -','')
                prod_str = prod_str.replace('Amazon Brand –','')
                prod_str = prod_str.replace('AmazonCommercial', 'Amazon Commercial')
                prod_str = prod_str.replace('find.', '')
                prod_str = prod_str.replace('365 EVERYDAY VALUE','365 Everyday Value,')
                prod_str = prod_str.replace('Fresh Brand –', 'Fresh Brand, ')
                prod_str = prod_str.replace('AMAZON', 'Amazon, ')
                prod_str = prod_str.replace('WHOLE FOODS MARKET', 'Whole Foods Market, ')
                prod_str = prod_str.strip()
        
                temp_str = "We sell " + prod_str + ". It is important to note that its Product ID is " + curr_id + ". "
                temp_str = temp_str.strip() + ". "
                self.item_id_dict[curr_id]['feature_fields']['product_name'] = prod_str
                llm_str += temp_str
        
            if 'brand' in self.item_id_dict[curr_id].keys():
                brand_str = self.item_id_dict[curr_id]['brand'][0]['value']
                temp_str = brand_str.strip()
                if temp_str != 'find.':
                    # Weird value of brand == 'find.'. Want to ignore this.
                    llm_str += "Its brand is " + temp_str + ". "
                    self.item_id_dict[curr_id]['feature_fields']['brand'] = temp_str
                
            self.item_id_dict[curr_id]['llm_str'] = llm_str
        return 

    def get_product_type_text(self):
        # Clean data for product type & construct 3rd sentence of 
        # Reference text.
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if 'product_type' in self.item_id_dict[curr_id].keys():
                prod_str = self.item_id_dict[curr_id]['product_type'][0]['value']
                temp_str = prod_str.replace('_',' ').lower()
                if temp_str == 'fashionearring':
                    temp_str = 'fashion earring'
                elif temp_str == 'fashionnecklacebraceletanklet':
                    temp_str = 'fashion necklace, bracelet or anklet'
                elif temp_str == 'fineearring':
                    temp_str = 'fine earring'
                elif temp_str == 'finenecklacebraceletanklet':
                    temp_str = 'fine necklace, bracelet or anklet'
                elif temp_str == 'fineother':
                    temp_str = 'fine other'
                elif temp_str == 'finering':
                    temp_str = 'fine ring'
                else:
                    pass
                llm_str = " This product type may be categorized as a " + temp_str + " product. "
                self.item_id_dict[curr_id]['feature_fields']['product_type'] = temp_str
                self.item_id_dict[curr_id]['llm_str'] += llm_str
        return 
    
    def get_product_desc_text(self):
        # Clean data for product type & construct 4th sentence of 
        # Reference text.
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if (('product_description' in self.item_id_dict[curr_id].keys()) and 
                (len (self.item_id_dict[curr_id]['product_description']) >0)):
                temp_str = self.item_id_dict[curr_id]['product_description'][0]['value']
                llm_str = " This product type may be described with this phrase: '" + temp_str + "'"
                self.item_id_dict[curr_id]['feature_fields']['product_description'] = temp_str
                self.item_id_dict[curr_id]['llm_str'] += llm_str
        return 
    
    def get_weight_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if 'item_weight' in self.item_id_dict[curr_id].keys():
                weight_dict = self.item_id_dict[curr_id]['item_weight'][0]
                llm_str = f"It weighs {weight_dict['value']:.2f} {weight_dict['unit']}"
                if 'normalized_value' in weight_dict.keys() and weight_dict['normalized_value']['unit'] != weight_dict['unit']:
                        llm_str += f", which is  equivalent to {weight_dict['normalized_value']['value']:.2f} {weight_dict['normalized_value']['unit']}. "
                else:
                    llm_str += ". "
                self.item_id_dict[curr_id]['feature_fields']['weight'] = f"{weight_dict['value']:.2f} {weight_dict['unit']}"
                self.item_id_dict[curr_id]['llm_str'] += llm_str
        return 

    def get_dimensions_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if 'item_dimensions' in self.item_id_dict[curr_id].keys():
                llm_str = ""
                dims_dict = self.item_id_dict[curr_id]['item_dimensions']
                if 'length' in dims_dict.keys():
                    llm_str += f"Its length is {dims_dict['length']['value']:.2f} {dims_dict['length']['unit']}"
                    if 'normalized_value' in dims_dict['length'].keys() and dims_dict['length']['normalized_value']['unit'] != dims_dict['length']['unit']:
                        llm_str += f", which is equivalent to {dims_dict['length']['normalized_value']['value']:.2f} {dims_dict['length']['normalized_value']['unit']}. "
                    else:
                        llm_str += ". "
            
                if 'width' in dims_dict.keys():
                    llm_str += f"Its width is {dims_dict['width']['value']:.2f} {dims_dict['width']['unit']}"
                    if 'normalized_value' in dims_dict['width'].keys() and dims_dict['width']['normalized_value']['unit'] != dims_dict['width']['unit']:
                        llm_str += f", which is equivalent to {dims_dict['width']['normalized_value']['value']:.2f} {dims_dict['width']['normalized_value']['unit']}. "
                    else:
                        llm_str += ". "

                if 'height' in dims_dict.keys():
                    llm_str += f"Its height is {dims_dict['height']['value']:.2f} {dims_dict['height']['unit']}"
                    if 'normalized_value' in dims_dict['height'].keys() and dims_dict['height']['normalized_value']['unit'] != dims_dict['height']['unit']:
                        llm_str += f", which is equivalent to {dims_dict['height']['normalized_value']['value']:.2f} {dims_dict['height']['normalized_value']['unit']}. "
                    else:
                        llm_str += ". "
                    

                dim_parts = []
                for dim_name in ['length', 'width', 'height']:
                    if dim_name in dims_dict:
                        dim_parts.append(
                            f"{dims_dict[dim_name]['value']:.2f} {dims_dict[dim_name]['unit']}"
                        )

                if dim_parts:
                    self.item_id_dict[curr_id]['feature_fields']['dimensions'] = " x ".join(dim_parts)
                self.item_id_dict[curr_id]['llm_str'] += llm_str

        return 
    
    def get_color_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if (('color' in self.item_id_dict[curr_id].keys()) and
                (len(self.item_id_dict[curr_id]['color'])>0)):

                llm_str = ""

                color_dict = self.item_id_dict[curr_id]['color'][0]
                llm_str += f"Its color may be described as {color_dict['value']}"

                if 'standardized_values' in color_dict.keys() and color_dict['standardized_values'][0] != color_dict['value']:
                    llm_str += f", or more simply {color_dict['standardized_values'][0]}. "
                else:
                    llm_str += ". "
                self.item_id_dict[curr_id]['feature_fields']['color'] = color_dict['value']
                self.item_id_dict[curr_id]['llm_str'] += llm_str
        return 

    def get_material_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            if 'material' in self.item_id_dict[curr_id].keys():
                llm_str = ""
                mat_str = ""
                mat_dicts = self.item_id_dict[curr_id]['material']
                for ctr, mat_dict in enumerate(mat_dicts):
                    if ctr == 0:
                        llm_str += f"It consists of {mat_dict['value']}"
                        mat_str += mat_dict['value']
                    else:
                        llm_str += f" and {mat_dict['value']}"
                        mat_str += f", {mat_dict['value']}"
                if len(llm_str.strip()) > 0:
                    llm_str += ". " 
                    self.item_id_dict[curr_id]['feature_fields']['material'] = mat_str
                    self.item_id_dict[curr_id]['llm_str'] += llm_str
        return 
    
    def get_node_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            cat_paths = []
            if (('node' in self.item_id_dict[curr_id].keys()) and 
                (self.item_id_dict[curr_id]['node'][0]['node_name'] is not None)):
                
                for ctr, node_dict in enumerate(self.item_id_dict[curr_id]['node']):
                    node_name = node_dict.get('node_name')
                    if isinstance(node_name, str) and len(node_name.split('/')) > 2:
                        cat_list = node_name.split('/')[2:]
                        if len(cat_paths) == 0:                   
                            llm_str += f"The categorical groups of this item, going from most general to least general are:\n"
                        else:
                            llm_str += f"Additionally, the categorical groups of this item from most general to least general could also be expressed as:\n"
                        cat_paths.append("/".join(cat_list))

                        
                        for ctr2 in range(len(cat_list)):
                            llm_str += f"    {str(ctr2+1)}. {cat_list[ctr2]}\n"
                            
                if len(llm_str) > 0:
                    self.item_id_dict[curr_id]['feature_fields']['categories'] = cat_paths
                    self.item_id_dict[curr_id]['llm_str'] += llm_str                  
        return 
    
    def get_fabric_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('fabric_type' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['fabric_type']))) > 0:
                llm_str += f"The fabric type is {self.item_id_dict[curr_id]['fabric_type'][0]['value']}. "
                self.item_id_dict[curr_id]['feature_fields']['fabric'] = self.item_id_dict[curr_id]['fabric_type'][0]['value']
            self.item_id_dict[curr_id]['llm_str'] += llm_str    
        return 

    def get_model_year_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('model_year' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['model_year']))) > 0:
                llm_str += f"The model year of this item is {str(self.item_id_dict[curr_id]['model_year'][0]['value'])}. "
                self.item_id_dict[curr_id]['feature_fields']['model_year'] = str(self.item_id_dict[curr_id]['model_year'][0]['value'])
                self.item_id_dict[curr_id]['llm_str'] += llm_str    
        return 
    
    def get_finish_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('finish_type' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['finish_type']))) > 0:
                llm_str += f"The finish type of this item is {str(self.item_id_dict[curr_id]['finish_type'][0]['value'])}. "
                self.item_id_dict[curr_id]['feature_fields']['finish'] = str(self.item_id_dict[curr_id]['finish_type'][0]['value'])
                self.item_id_dict[curr_id]['llm_str'] += llm_str   
        return 
    
    def get_pattern_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            pattern_list = []
            if (('pattern' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['pattern']))) > 0:
                for pattern_dict in self.item_id_dict[curr_id]['pattern']:
                    if isinstance(pattern_dict['value'], str) and len(pattern_dict['value']) > 2:
                        if len(pattern_list) == 0:                    
                            llm_str += f"This item has a pattern that can be characterized as '{pattern_dict['value']}'"
                        else:
                            llm_str += f"Additionally, this item's pattern can be characterized as '{pattern_dict['value']}'"
                        pattern_list.append(pattern_dict['value'])

                if len(llm_str) > 0:
                    llm_str += ". "
                    self.item_id_dict[curr_id]['feature_fields']['pattern'] = pattern_list
                    self.item_id_dict[curr_id]['llm_str'] += llm_str     
        return
    
    def get_shape_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('item_shape' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['item_shape']))) > 0:
                llm_str += f"This item's shape may be described as '{self.item_id_dict[curr_id]['item_shape'][0]['value']}'. "
                self.item_id_dict[curr_id]['feature_fields']['shape'] = self.item_id_dict[curr_id]['item_shape'][0]['value']
                self.item_id_dict[curr_id]['llm_str'] += llm_str    
        return 
    
    def get_bullet_point_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('bullet_point' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['bullet_point']))) > 0:
        
                for ctr, bullet_dict in enumerate(self.item_id_dict[curr_id]['bullet_point']):
        
                    if isinstance(bullet_dict['value'], str) and len(bullet_dict['value']) > 2:
                        if ctr == 0:                    
                            llm_str += f"There are several bullet points associated with this item:\n"
                            llm_str += f"    {str(ctr+1)}. {bullet_dict['value']}\n"
                        else:
                            llm_str += f"    {str(ctr+1)}. {bullet_dict['value']}\n"
                if len(llm_str) > 0:
                    self.item_id_dict[curr_id]['llm_str'] += llm_str 
        return 
    
    def get_main_img_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('main_image_id' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['main_image_id']))) > 0:
                llm_str = f"The main image id associated with this item is {self.item_id_dict[curr_id]['main_image_id']}. "
            if len(llm_str) > 0:
                self.item_id_dict[curr_id]['llm_str'] += llm_str   
        return 
    
    def get_other_img_text(self):
        item_list = sorted(self.item_id_dict.keys())
        for curr_id in item_list:
            llm_str = ""
            if (('other_image_id' in self.item_id_dict[curr_id].keys()) and 
                (len(self.item_id_dict[curr_id]['other_image_id']))) > 0:
                temp_str = ", ".join(self.item_id_dict[curr_id]['other_image_id'])
                llm_str = f"Additional images associated with this item are {temp_str}. "
                if len(temp_str) > 0:
                    self.item_id_dict[curr_id]['llm_str'] += llm_str   
        return 

    def make_blurb_dict(self):
        '''Make dictionary of blurbs'''
        out_dict = defaultdict(dict)
        for curr_id in self.item_id_dict.keys():
            if (
                'item_name' not in self.item_id_dict[curr_id]
                or not self.item_id_dict[curr_id]['item_name']
                or 'value' not in self.item_id_dict[curr_id]['item_name'][0]
            ):
                raise ValueError(f"Missing item_name for product {curr_id}")

            out_dict[curr_id]['llm_str'] = self.item_id_dict[curr_id]['llm_str']
            out_dict[curr_id]['item_name'] = self.item_id_dict[curr_id]['item_name'][0]['value']
            out_dict[curr_id]['feature_fields'] = self.item_id_dict[curr_id]['feature_fields']

            if 'main_image_id' in self.item_id_dict[curr_id].keys():
                out_dict[curr_id]['main_image_id'] = self.item_id_dict[curr_id]['main_image_id']
            else:
                out_dict[curr_id]['main_image_id'] = None

            if 'other_image_id' in self.item_id_dict[curr_id].keys():
                out_dict[curr_id]['other_image_id'] = self.item_id_dict[curr_id]['other_image_id']
            else:
                out_dict[curr_id]['other_image_id'] = None

        self.blurb_dict = out_dict
        return 

    def save_full_blurb_dict(self, output_dir, output_json, full_blurb_dict):
        '''Save full blurb dictionary to file'''
        # Create output directory if it doesn't exist
        output_path = Path(output_dir) 
        if not output_path.exists():
            output_path.mkdir(parents=True)
            print(f"Created directory {output_path.resolve()}")
        
        # Save the full dictionary
        #print(f"Saving full blurb dictionary to {output_dir}/{output_json}")
        with open(Path(output_path) / Path(output_json), "w", encoding="utf-8") as f:
            json.dump(full_blurb_dict, f, indent=4)

        return

       
            
if __name__ == '__main__':      
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default ='./data')
    parser.add_argument('--listing_dir', type=str, default='abo-listings/listings/metadata')
    parser.add_argument('--country', type=str, default='US')
    parser.add_argument('--language', type=str, default='en_US')
    parser.add_argument('--output_dir', type=str, default='./EDA/product_blurbs')
    parser.add_argument('--config_file', type=str, default='./EDA/config.ini')
    parser.add_argument('-o','--output_json', type=str, default='combined_blurb_dict.json')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-z', '--zipped', action='store_true')
    args = parser.parse_args()

    # Unzip files
    if args.zipped:
        for curr_file in [x for x in Path(args.root_dir, args.listing_dir).iterdir() 
                          if x.suffix == '.gz']:
            print(f"Found {curr_file}, {curr_file.suffix}")

            with gzip.open(curr_file, 'rb') as f_in:
                json_file = curr_file.parent / Path(curr_file.name[:-3])
                with open(json_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


    full_blurb_dict = dict()
    tot=0
    json_files = [x for x in Path(args.root_dir, args.listing_dir).iterdir()
                  if x.suffix == '.json']
    if not json_files:
        raise FileNotFoundError(
            f"No .json listing files found in {Path(args.root_dir, args.listing_dir)}"
        )
    for curr_file in json_files:
        print(f"\nFound {curr_file}")

        dg = DescriptionGenerator(args.root_dir, args.listing_dir, 
                                  curr_file.name, 
                                  args.country, args.language,
                                  args.config_file)
        dg.make_data_dict()
        dg.get_all_text()
        dg.make_blurb_dict()
        
        # Track number of blurbs and add to full blurb dictionary
        tot+=len(dg.blurb_dict)

        if args.test:
            ctr = 0
            for item_id, meta_info_dicts in dg.item_id_dict.items():
                print(meta_info_dicts['llm_str'])
                print()
                ctr += 1
                if ctr > 5:
                    break
            
        else:
            dup_keys = set(full_blurb_dict.keys()).intersection(set(dg.blurb_dict.keys()))
            print(f"\n  Found {len(dup_keys)} duplicate keys")
            full_blurb_dict = full_blurb_dict | dg.blurb_dict
            print(f"  Current Expected total number of blurbs: {tot}")
            print(f"  Current Actual total number of blurbs: {len(full_blurb_dict)}")



    # Save full blurb dictionary
    if not args.test:
        # Preprocess documents
        preprocessor = Preprocessor(full_blurb_dict)
        preprocessor.preprocess_documents()

        dg.save_full_blurb_dict(args.output_dir, args.output_json, full_blurb_dict)
        print(f"\nSaved full blurb dictionary to {args.output_dir}/{args.output_json}")
        print(f"Final Expected total number of blurbs: {tot}")
        print(f"Final Actual total number of blurbs: {len(full_blurb_dict)}")

