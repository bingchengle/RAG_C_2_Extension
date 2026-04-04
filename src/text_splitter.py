import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables."""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        '''
        入参：parsed_report.get('tables', []) —— 从解析后的报告字典中获取 tables 字段的值，如果没有该字段则返回空列表，避免报错；
        '''

        for page in file_content['content']['pages']:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        '''
        string: str：要计算 token 的目标字符串（类型注解标注为字符串）
        encoding_name="o200k_base"：编码方案名称（可选参数，默认值为 o200k_base），不同的编码对应不同的大语言模型
        '''
        encoding = tiktoken.get_encoding(encoding_name)
        '''
        tiktoken.get_encoding(encoding_name)：根据指定的编码名称（如 o200k_base）获取对应的编码规则对象
        赋值：将编码规则对象存入变量 encoding，后续用它来编码字符串
        '''

        tokens = encoding.encode(string)
        '''
        encoding.encode(string)：使用上面获取的编码规则，将输入的字符串转换为一串整数（每个整数代表一个 token）
        '''
        token_count = len(tokens)

        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            #.from_tiktoken_encoder()：通过 tiktoken 编码器初始化分割器（而非按字符数分割），确保分割依据是token 数（符合 LLM 的处理逻辑）
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        #初始化空列表 chunks_with_meta，用于存储带元信息的文本块（纯文本 + 页码 + token 数）
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):

        all_report_paths = list(all_report_dir.glob("*.json"))
        '''
        all_report_dir.glob("*.json")：遍历 all_report_dir 目录下所有后缀为 .json 的文件，返回一个可迭代的路径对象
        list(...)：将可迭代对象转换为列表，每个元素是一个 Path 类型的文件路径
        '''

        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                #serialized_tables_dir / report_path.name：拼接表格目录和文件名，得到对应的表格文件路径（例如：/tables/report1.json）
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            '''
            output_dir.mkdir()：创建输出目录
            parents=True：如果父目录不存在，自动创建（例如：/output/reports 不存在时，会先创建 /output 再创建 /output/reports）
            exist_ok=True：如果目录已存在，不报错（避免重复创建的异常）
            '''
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                '''
                updated_report：待写入的处理后报告数据
                file：写入的文件句柄
                indent=2：JSON 字符串缩进 2 个空格，保证可读性
                ensure_ascii=False：允许写入非 ASCII 字符（如中文），避免中文被转义
                '''

        print(f"Split {len(all_report_paths)} files")
