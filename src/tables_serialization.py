import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from src.api_requests import BaseOpenaiProcessor, AsyncOpenaiProcessor
import tiktoken
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

message_queue = Queue()
#临时存放待输出的日志信息，后续可单独消费队列中的日志，避免和 tqdm 进度条的输出互相干扰。

class TqdmLoggingHandler(logging.Handler):
    '''
    继承自 logging.Handler（Python 日志模块的基础处理器类）
    替换默认的日志输出方式（比如 StreamHandler 直接打印到控制台），改为「先存队列，再统一处理」
    '''
    def emit(self, record):
        '''
        emit 是 logging.Handler 的核心抽象方法，日志模块会在产生日志时自动调用这个方法，
        参数 record 是日志记录对象（包含日志级别、内容、时间、调用栈等信息）。
        '''
        try:
            msg = self.format(record)
            '''
            调用 self.format(record) 按照日志配置的格式
            （比如 %(asctime)s - %(levelname)s - %(message)s）将 record 转换成可读的字符串 msg
            '''
            message_queue.put((record.levelno, msg))
            '''
            往日志队列中存入(record.levelno, msg)元组
            record.levelno：日志的「级别编号」。record 是日志对象，levelno 是它的属性，是一个整数，代表日志的严重程度
            debug10，info20，warning30，error40，critical50
            '''
        except Exception:
            self.handleError(record)
            '''
            捕获 emit 方法执行过程中可能出现的异常（比如格式化失败、队列写入失败），
            调用 handleError（父类 logging.Handler 提供的默认错误处理方法）记录错误，避免日志处理器崩溃导致整个程序挂掉。
            '''

def process_messages():
    '''
    打印日志函数
    '''
    while not message_queue.empty():
        '''
        判断是否取完日志,取完则返回true后停止函数
        '''
        level, msg = message_queue.get_nowait()
        '''
        从之前日志元组中取出日志编号和文本并赋值给level和msg
        message_queue.get_nowait()：队列的非阻塞取值方法
        get()：阻塞式：如果队列为空，会一直等待，直到队列有数据才返回（适合单独线程消费）
        get_nowait()：非阻塞式：如果队列为空，直接抛出 queue.Empty 异常（适合循环内快速取值）
        '''
        tqdm.write(msg)
        '''
        安全打印日志：tqdm.write()：tqdm 库提供的专用输出方法
        print(msg):会打断 tqdm 进度条，导致进度条错位、多行混乱——简单，适合无进度条的场景
        tqdm.write(msg)：自动避开进度条输出区域，不干扰进度条显示——专为「进度条 + 日志」场景设计
        '''

class TableSerializer(BaseOpenaiProcessor):
    def __init__(self, preserve_temp_files: bool = True):
        '''
        preserve_temp_files: bool = True：可选参数，默认 True，
        表示「是否保留处理过程中生成的临时文件」
        '''
        super().__init__()
        self.preserve_temp_files = preserve_temp_files
        os.makedirs('./temp', exist_ok=True)
        '''
        os.makedirs：创建文件夹（支持多级目录）
        './temp'：临时文件夹路径（当前目录下的 temp 文件夹）
        exist_ok=True：关键参数，表示「如果文件夹已存在，不报错」（避免重复创建导致异常）
        '''
        
        self.logger = logging.getLogger('TableSerializer')
        '''
        创建一个名为 TableSerializer 的日志器（logger）
        隔离日志上下文，比如 TableSerializer 的日志只打自己的内容，不会和其他模块（比如文本分片、API 调用）的日志混在一起，方便排查问题
        '''

        self.logger.setLevel(logging.INFO)
        '''
        只输出 INFO 及以上级别的日志
        '''
        self.logger.handlers.clear()
        '''
        清除该日志器上已绑定的所有处理器
        目的：避免日志重复输出（比如既用自定义的 TqdmLoggingHandler，又用默认的控制台处理器，导致一条日志打两次）
        '''
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(handler)
        '''
        handler.setFormatter(...)：设置日志格式为 级别: 消息
        self.logger.addHandler(handler)：把这个处理器绑定到当前日志器上，后续日志会通过这个处理器存入队列
        '''
        self.logger.propagate = False
        '''
        propagate 默认是 True，表示日志会向上传播到父日志器（比如根日志器）
        设置为 False：确保 TableSerializer 的日志只通过自己的 TqdmLoggingHandler 处理，不会被其他日志器重复输出。
        '''

    def _get_table_context(self, json_report, target_table_index):
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        page_num = table_info["page"]
        
        page_content = next(
            (page["content"] for page in json_report["content"] if page["page"] == page_num),
            []
        )
        
        if not page_content:
            self.logger.warning(f"Page {page_num} not found for table {target_table_index}")
            return "", ""
        '''
        如果 page_content 是空列表（没找到对应页码的内容），通过子类的 self.logger 输出警告日志
        '''

        # Find position of target table in page_content
        current_table_position = -1
        #仅是做标记，若没有找到break后返回-1，直观表示没有找到
        for i, block in enumerate(page_content):
            if block["type"] == "table" and block.get("table_id") == target_table_index:
                current_table_position = i
                break
                #定位目标table的位置

        # Find position of previous table if exists
        previous_table_position = -1
        for i in range(current_table_position-1, -1, -1):
            if page_content[i]["type"] == "table":
                previous_table_position = i
                break
                #向前找类型是table的id，记为current_table_position并停止

        # Find position of next table if exists
        next_table_position = -1
        for i in range(current_table_position + 1, len(page_content)):
            if page_content[i]["type"] == "table":
                next_table_position = i
                break
                #向后遍历类型是table的id，记为previous_table_position并停止

        # Get blocks above current table
        start_position = previous_table_position + 1 if previous_table_position != -1 else 0
        '''如果不是第一个table，取值范围就是上一个table到当前table中间的文本，若没有前一个table就直接从开头开始取到目标table'''
        context_before = page_content[start_position:current_table_position]
        '''table上方的文本通常是「标题 / 说明 / 前置条件」，是理解表格的核心，且不会无限制长,因此全取'''

        # Get blocks after current table
        context_after = []
        if next_table_position == -1:
            # If no next table, take up to 3 blocks until end of page
            context_after = page_content[current_table_position + 1:current_table_position + 4]
        else:
            # If next table exists, take up to 3 blocks before it
            blocks_between = next_table_position - (current_table_position + 1)
            if blocks_between > 3:
                context_after = page_content[current_table_position + 1:current_table_position + 4]
            elif blocks_between > 1:
                context_after = page_content[current_table_position + 1:current_table_position + blocks_between]
                #table下方没有table就默认去三个block，如果有，block大于3也取3，小于3就取完
                #表格下方的文本可能是「备注 / 其他表格的说明 / 无关内容」，过长会浪费 Token，需限制长度；

        context_before = "\n".join(block.get("text", "") for block in context_before if "text" in block)
        context_after = "\n".join(block.get("text", "") for block in context_after if "text" in block)
        #仅保留文本blocks

        return context_before, context_after

    def _send_serialization_request(self, table, context_before, context_after):
        user_prompt = ""
        '''创建空字符串 user_prompt，用于拼接最终发给大模型的用户指令'''

        if context_before:
            user_prompt += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
        '''
        if context_before:：判断是否有表格上文（非空字符串），无则跳过
        格式化拼接：用自然语言说明「这是表格上方的相关文本」，降低模型理解成本；用三重引号 """ 包裹上下文文本，避免文本中的换行 / 符号破坏提示词格式；;
        末尾加 \n\n 换行，保证提示词结构清晰（上下文和表格之间空行分隔）
        '''

        user_prompt += f'Here is a table in HTML format:\n"""{table}"""'
        '''固定文本说明「这是 HTML 格式的表格」，明确告诉模型输入的表格格式；'''

        if context_after:
            user_prompt += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'

        '''
        类属性会在「类定义阶段」被优先加载（不管写在类内哪个位置）
        方法内的代码要等到「方法执行阶段」才运行，此时类属性早已加载完成
        '''
        system_prompt = TableSerialization.system_prompt
        reponse_schema = TableSerialization.TableBlocksCollection
        '''
        TableBlocksCollection：类的类属性（通常是 Pydantic 模型），定义了表格解析结果的结构化格式（比如包含 table_id、rows、columns、cells 等字段）；
        作用：强制模型输出符合预设格式的结果，避免自由文本解析的格式混乱，方便后续代码处理。
        '''

        answer_dict = self.send_message(
            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=system_prompt,
            human_content=user_prompt,
            is_structured=True,
            response_format=reponse_schema
        )
        '''返回值 answer_dict：模型解析后的结构化表格数据（字典格式，符合 TableBlocksCollection 规则）'''

        input_message = user_prompt + system_prompt + str(reponse_schema.schema())
        input_tokens = self.count_tokens(input_message)
        output_tokens = self.count_tokens(str(answer_dict))

        result = answer_dict
        return result
    
    def _serialize_table(self, json_report: dict, target_table_index: int) -> dict:

        # Get the context surrounding the table
        context_before, context_after = self._get_table_context(json_report, target_table_index)
        
        # Get the table content
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        table_content = table_info["html"]
        
        # Serialize the table with its context
        result = self._send_serialization_request(
            table=table_content,
            context_before=context_before,
            context_after=context_after
        )
        
        return result

    def serialize_tables(self, json_report: dict) -> dict:
        """Process all tables in the report and add serialization results to each table's info"""
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            
            # Get serialization results for current table
            serialization_result = self._serialize_table(
                json_report=json_report,
                target_table_index=table_index
            )
            
            # Add serialization results to the table info
            table["serialized"] = serialization_result
        
        return json_report

    async def async_serialize_tables(
            #async def：标记为异步方法（需通过 await 调用），核心目标是并行调用 OpenAI API，提升批量表格解析效率

        self, 
        json_report: dict,
        requests_filepath: str = './temp_async_llm_requests.jsonl',
        results_filepath: str = './temp_async_llm_results.jsonl'
            #requests_filepath/results_filepath：异步请求 / 结果的临时文件路径（默认值），用于调试 / 留存数据
    ) -> dict:
        """Process all tables in the report asynchronously"""
        '''整体作用是利用异步批量处理，将table序列化，然后建立一个字典，遍历回填tableid和序列化table'''
        queries = []
        table_indices = []
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            table_indices.append(table_index)
            
            context_before, context_after = self._get_table_context(json_report, table_index)
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            table_content = table_info["html"]
            
            # Construct the query
            query = ""
            if context_before:
                query += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
            query += f'Here is a table in HTML format:\n"""{table_content}"""'
            if context_after:
                query += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
            
            queries.append(query)

        results = await AsyncOpenaiProcessor().process_structured_ouputs_requests(
            #await：异步等待 API 调用完成（这是异步方法的核心关键字，暂停当前方法直到 API 返回结果，不阻塞其他任务）
            #AsyncOpenaiProcessor()：异步 OpenAI 处理器实例（区别于同步的 BaseOpenaiProcessor）
            #process_structured_ouputs_requests：异步批量处理结构化输出请求的核心方法

            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=TableSerialization.system_prompt,
            queries=queries,
            response_format=TableSerialization.TableBlocksCollection,
            preserve_requests=False,
            preserve_results=False,
            logging_level=20,
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
        )

        # Add results back to json_report
        for table_index, result in zip(table_indices, results):
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            
            new_table = {}
            for key, value in table_info.items():
                new_table[key] = value
                if key == "html":
                    new_table["serialized"] = result["answer"]
            
            for i, table in enumerate(json_report["tables"]):
                if table["table_id"] == table_index:
                    json_report["tables"][i] = new_table
                    '''
                    
                    '''

        return json_report

    def process_file(self, json_path: Path) -> None:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_report = json.load(f)
            '''
            'r'：只读模式，encoding='utf-8'：指定 UTF-8 编码，避免中文乱码；
            json.load(f)：将文件中的 JSON 字符串解析为 Python 字典
            '''

            thread_id = threading.get_ident()
            '''threading.get_ident()：获取当前执行该方法的线程 ID（整数）'''

            requests_filepath = f'./temp/async_llm_requests_{thread_id}.jsonl'
            results_filepath = f'./temp/async_llm_results_{thread_id}.jsonl'
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            '''
            asyncio.new_event_loop()：创建一个新的异步事件循环（loop）,类似于实例化对象去调用函数，没有创建就无法进行上述异步编程
            asyncio.set_event_loop(loop)：将该循环设置为当前线程的默认事件循环；
            '''

            try:
                updated_report = loop.run_until_complete(self.async_serialize_tables(
                    json_report,
                    requests_filepath=requests_filepath,
                    results_filepath=results_filepath
                ))
                '''
                loop.run_until_complete(...)：启动事件循环，等待异步方法 async_serialize_tables 执行完成，返回结果
                '''

            finally:
                loop.close()
                try:
                    os.remove(requests_filepath)
                    os.remove(results_filepath)
                except FileNotFoundError:
                    pass
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
                '''
                'w'：写入模式（覆盖原文件）
                json.dump(...)：将处理后的 updated_report 字典转换为 JSON 字符串，写入文件
                indent=2：JSON 字符串格式化缩进 2 个空格，便于人工阅读
                ensure_ascii=False：允许输出非 ASCII 字符（如中文），避免中文被转义为 '/uXXXX'格式；
                '''
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON Error in %s: %s", json_path.name, str(e))
            '''self.logger.error(...)：记录错误日志，包含文件名和具体错误信息'''
            raise
        except Exception as e:
            #Exception as e：捕获除 JSON 解析外的所有异常（文件 IO、异步调用、权限问题等）
            self.logger.error("Error processing %s: %s", json_path.name, str(e))
            raise

    def process_directory_parallel(self, input_dir: Path, max_workers: int = 5):
        """Process JSON files in parallel using thread pool.
        
        Args:
            input_dir: 待处理的目录路径
            max_workers: 线程池最大线程数（默认 5），控制并行处理的文件数量
        """
        self.logger.info("Starting parallel table serialization...")
        '''记录 INFO 级日志，提示 “开始并行表格序列化”'''
        
        json_files = list(input_dir.glob("*.json"))
        '''
        glob("*.json")：Path对象的内置方法，作用是「查找目录下匹配指定规则的文件」
        "*.json"：匹配规则，*是通配符，代表 “任意字符”，整体意思是 “所有后缀为.json的文件”
        生成的地址迭代器转列表的原因：
        1、需统计文件总数，原迭代器无法用len；
        2、需朝服使用，迭代器只能遍历一次，再次遍历会为空
        3、便于打印，迭代器只能看到id
        '''
        
        if not json_files:
            self.logger.warning("No JSON files found in %s", input_dir)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            '''
            ThreadPoolExecutor：Python 标准库的线程池管理器，用于创建固定数量的线程；
            with 上下文管理器：自动管理线程池的创建 / 关闭，避免线程泄露；
            '''
            with tqdm(
                total=len(json_files),
                desc="Processing files",
                mininterval=1.0,
                maxinterval=5.0,
                smoothing=0.3
            ) as pbar:
                '''
                tqdm：Python 常用的进度条库，用于可视化展示文件处理进度；
                desc="Processing files"：进度条描述文字（显示 “Processing files”）；
                mininterval=1.0/maxinterval=5.0：进度条更新的最小 / 最大间隔（1-5 秒），避免频繁刷新；
                smoothing=0.3：进度条平滑度，让进度更新更自然
                as pbar：进度条实例，后续通过pbar.update(1)更新进度；
                '''
                futures = []
                for json_file in json_files:
                    future = executor.submit(self.process_file, json_file)
                    '''
                    executor.submit(...)：将self.process_file(json_file)任务提交到线程池；
                    返回值future：Future 对象，代表 “待完成的任务”，可用于后续监控任务状态.向线程池分配任务，如果线程都在忙，就排队
                    '''
                    future.add_done_callback(lambda p: pbar.update(1))
                    '''
                    future.add_done_callback(...)：为 Future 对象添加 “任务完成时的回调函数”
                    lambda p: pbar.update(1)：匿名函数，任务完成时调用pbar.update(1)，进度条前进 1 格
                    '''
                    futures.append(future)
                
                while futures:
                    process_messages()
                    
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(str(e))
                    
                    for future in done_futures:
                        futures.remove(future)
                    '''将已处理的 Future 对象从futures列表中移除,当futures列表为空时，循环结束（所有任务完成）'''
                    time.sleep(0.1)
                    '''每次循环休眠 0.1 秒，避免循环过于频繁导致 CPU 占用过高'''

        process_messages()
        self.logger.info("Table serialization completed!")
        '''记录 INFO 级日志，提示 “表格序列化完成”'''


class TableSerialization:
        
    system_prompt = (
        "You are a table serialization agent.\n"
        "Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n"
        "These blocks must be totally context-independent because they will be used as separate chunk to populate database."
    )

    class SerializedInformationBlock(BaseModel):
        "A single self-contained information block enriched with comprehensive context"

        subject_core_entity: str = Field(description="A primary focus of what this block is about. Usually located in a row header. If one row in the table doesn't make sense without neighboring rows, you can merge information from neighboring rows into one block")
        information_block: str = Field(description=(
    "Detailed information about the chosen core subject from tables and additional texts. Information SHOULD include:\n"
    "1. All related header information\n"
    "2. All related units and their descriptions\n"
    "    2.1. If header is Total, always write additional context about what this total represents in this block!\n"
    "3. All additional info for context enrichment to make ensure complete context-independency if it present in whole table. This can include:\n"
    "    - The name of the table\n"
    "    - Additional footnotes\n"
    "    - The currency used\n"
    "    - The way amounts are presented\n"
    "    - Anything else that can make context even slightly richer\n"
    "SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!"
    ))

    class TableBlocksCollection(BaseModel):
        """Collection of serialized table blocks with their core entities and header relationships"""

        subject_core_entities_list: List[str] = Field(
            description="A complete list of core entities. Keep in mind, empty headers are possible - they should also be interpreted and listed (Usually it's a total or something similar). In most cases each row header represents a core entity")
        relevant_headers_list: List[str] = Field(description="A list of ALL headers relevant to the subject. These headers will serve as keys in each information block. In most cases each column header represents a core entity")
        information_blocks: List["TableSerialization.SerializedInformationBlock"] = Field(description="Complete list of fully described context-independent information blocks")
