from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import json
import requests
from openai import OpenAI
import re
from datetime import datetime
import os
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 请更改为安全的密钥

# DeepSeek API配置
client = OpenAI(
    api_key="sk-cec51cae7ee442da9c8f30fe4f26cc11", 
    base_url="https://api.deepseek.com"
)

# 全局变量存储数据
patient_data = {}
pubmed_articles = {}

# ICD9诊断代码到中文的完整映射
ICD9_DIAGNOSIS_MAPPING = {
    '99591': '感染和炎症反应',
    '99662': '严重脓毒症',
    '5672': '膀胱疾病',
    '40391': '高血压性慢性肾病',
    '42731': '心房颤动',
    '4280': '充血性心力衰竭',
    '4241': '急性心肌梗死',
    '4240': '冠心病',
    '2874': '血小板减少症',
    '03819': '其他脓毒血症',
    '7850': '心脏杂音',
    'E8791': '意外跌落',
    'V090': '感染筛查',
    '56211': '膀胱炎',
    '28529': '贫血',
    '25000': '2型糖尿病',
    'V5867': '长期使用抗凝剂',
    'E9342': '药物不良反应',
    '41401': '冠状动脉粥样硬化',
    '2749': '痛风',
    '3051': '谵妄',
    '570': '急性肝坏死',
    '07030': '病毒性肝炎',
    '07054': '慢性肝炎',
    '30401': '阿片类药物依赖',
    '2875': '血小板功能障碍',
    '2760': '酸碱平衡紊乱',
    '0389': '脓毒血症',
    '41071': '亚急性细菌性心内膜炎',
    '78551': '恶心呕吐',
    '486': '肺炎',
    '20280': '淋巴结肿大',
    '4582': '低血压',
    '2724': '低钠血症',
    '81201': '颈椎骨折',
    '4928': '呼吸衰竭',
    '8028': '胸部闭合性损伤',
    '8024': '胸部开放性伤口',
    '99812': '术后感染',
    '41511': '主动脉瓣狭窄',
    '2851': '急性失血性贫血',
    'E8859': '意外事故',
    'E8788': '其他意外',
    'V1259': '个人恶性肿瘤病史',
    '4019': '原发性高血压'
}

# ICD9手术操作代码到中文的映射
ICD9_PROCEDURE_MAPPING = {
    '9749': '呼吸机辅助通气',
    '5491': '膀胱切开术',
    '3895': '静脉导管置入',
    '3995': '血液透析',
    '3893': '动脉导管置入',
    '9907': '重症监护',
    '14': '皮肤切开术',
    '9915': '重症监护',
    '3891': '动脉置管',
    '8181': '脊柱融合术',
    '9904': '输血'
}

def translate_icd9_codes(icd9_codes_str, is_procedure=False):
    """将ICD9代码转换为中文诊断/操作名称"""
    if pd.isna(icd9_codes_str) or not icd9_codes_str:
        return []
    
    mapping = ICD9_PROCEDURE_MAPPING if is_procedure else ICD9_DIAGNOSIS_MAPPING
    
    # 处理多种分隔符
    codes = []
    if isinstance(icd9_codes_str, str):
        # 处理分号、逗号等分隔符
        code_list = re.split(r'[;,，；\s]+', str(icd9_codes_str))
        codes = [code.strip() for code in code_list if code.strip()]
    
    translated = []
    for code in codes:
        # 清理代码，移除非数字字符（除了E和V开头的代码）
        clean_code = code.strip()
        if clean_code.startswith(('E', 'V')):
            # 保持E和V开头的代码格式
            pass
        else:
            # 移除前导零和非数字字符
            clean_code = re.sub(r'^0+', '', clean_code)
            clean_code = re.sub(r'[^\dEV]', '', clean_code)
        
        if clean_code in mapping:
            translated.append(f"{mapping[clean_code]} ({clean_code})")
        elif clean_code:
            translated.append(f"诊断代码: {clean_code}")
    
    return translated[:10]  # 限制显示数量

def load_patient_data():
    """加载患者数据"""
    global patient_data
    try:
        df = pd.read_csv('structured_emr_rag.csv')
        for index, row in df.iterrows():
            patient_id = str(row['row_id'])
            
            # 使用新的ICD9翻译函数
            diagnosis_codes_translated = translate_icd9_codes(row.get('icd9_code', ''), is_procedure=False)
            procedure_codes_translated = translate_icd9_codes(row.get('icd9_code_proc', ''), is_procedure=True)
            
            patient_data[patient_id] = {
                'subject_id': row['subject_id'],
                'admission_time': row['admittime'],
                'discharge_time': row['dischtime'],
                'diagnosis_codes_raw': str(row.get('icd9_code', '')).split(';') if pd.notna(row.get('icd9_code')) else [],
                'diagnosis_codes_translated': diagnosis_codes_translated,
                'procedure_codes_raw': str(row.get('icd9_code_proc', '')).split(';') if pd.notna(row.get('icd9_code_proc')) else [],
                'procedure_codes_translated': procedure_codes_translated,
                'medications': str(row.get('drug', '')).split(';') if pd.notna(row.get('drug')) else [],
                'structured_report': str(row['structured_report_rag']) if pd.notna(row['structured_report_rag']) else ''
            }
        print(f"成功加载 {len(patient_data)} 个患者数据")
        return True
    except Exception as e:
        print(f"加载患者数据失败: {e}")
        return False

def extract_pubmed_from_csv():
    """从CSV文件中提取PubMed文献信息"""
    global pubmed_articles
    try:
        df = pd.read_csv('structured_emr_rag.csv')
        for index, row in df.iterrows():
            if pd.notna(row['structured_report_rag']):
                # 提取PMID
                pmids = re.findall(r'PMID:\s*(\d+)', str(row['structured_report_rag']))
                for pmid in pmids:
                    if pmid not in pubmed_articles:
                        # 提取文献摘要
                        pattern = f'PMID: {pmid}.*?摘要: (.*?)(?=PMID:|DOI:|$)'
                        match = re.search(pattern, str(row['structured_report_rag']), re.DOTALL)
                        if match:
                            pubmed_articles[pmid] = {
                                'pmid': pmid,
                                'abstract': match.group(1).strip(),
                                'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                            }
        print(f"成功提取 {len(pubmed_articles)} 篇PubMed文献")
        return True
    except Exception as e:
        print(f"提取PubMed文献失败: {e}")
        return False

def get_medical_advice(patient_id, user_question):
    """使用DeepSeek API获取医疗建议"""
    try:
        # 获取患者信息
        patient_info = patient_data.get(patient_id, {})
        
        # 构建上下文
        context = f"""
        患者ID: {patient_id}
        入院时间: {patient_info.get('admission_time', '未知')}
        出院时间: {patient_info.get('discharge_time', '未知')}
        诊断信息: {', '.join(patient_info.get('diagnosis_codes_translated', []))}
        治疗操作: {', '.join(patient_info.get('procedure_codes_translated', []))}
        用药记录: {', '.join(patient_info.get('medications', [])[:10])}
        
        详细病历信息:
        {patient_info.get('structured_report', '')[:1000]}
        
        相关医学文献库:
        """
        
        # 添加相关文献信息
        for pmid, article in list(pubmed_articles.items())[:3]:  # 限制文献数量
            context += f"\nPMID: {pmid}\n摘要: {article['abstract'][:300]}...\n"
        
        # 构建提示词
        system_prompt = """
        你是一个专业的医疗AI助手。请基于患者的病历信息和相关医学文献，为患者提供专业、准确的医疗建议。
        
        注意事项：
        1. 始终强调患者应咨询专业医生获取个性化建议
        2. 基于患者的具体病历信息提供针对性建议
        3. 结合相关医学文献提供循证医学支持
        4. 用简洁易懂的中文回答
        5. 不能替代专业医疗诊断和治疗
        """
        
        user_prompt = f"""
        患者信息：
        {context}
        
        患者问题：{user_question}
        
        请基于以上信息为患者提供专业的医疗建议。
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"API调用失败: {e}")
        return "抱歉，系统暂时无法提供服务，请稍后再试或咨询专业医生。"

@app.route('/')
def index():
    """主页面"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"渲染模板失败: {e}")
        # 如果模板不存在，返回简单的HTML
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Web Mediverse</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
                .button { display: inline-block; margin: 20px; padding: 15px 30px; 
                         background: #007bff; color: white; text-decoration: none; 
                         border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🏥 Web Mediverse</h1>
            <p>智能医疗对话平台</p>
            <a href="/doctor_login" class="button">👨‍⚕️ 医生入口</a>
            <a href="/patient_login" class="button">👤 患者入口</a>
            <p style="color: red;">注意：templates文件夹缺失，请创建并添加模板文件</p>
        </body>
        </html>
        """

@app.route('/doctor_login')
def doctor_login():
    """医生登录页面"""
    try:
        return render_template('doctor_login.html')
    except:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>医生登录</title></head>
        <body style="font-family: Arial; margin: 40px; text-align: center;">
            <h2>👨‍⚕️ 医生登录</h2>
            <form method="POST" action="/verify_doctor">
                <p>测试验证码：<strong>123</strong></p>
                <input type="password" name="verification_code" placeholder="请输入验证码" required>
                <br><br>
                <button type="submit">登录</button>
            </form>
            <a href="/">← 返回主页</a>
        </body>
        </html>
        """

@app.route('/patient_login')
def patient_login():
    """患者登录页面"""
    try:
        return render_template('patient_login.html')
    except:
        available_ids = list(patient_data.keys())[:5]
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>患者登录</title></head>
        <body style="font-family: Arial; margin: 40px; text-align: center;">
            <h2>👤 患者登录</h2>
            <form method="POST" action="/verify_patient">
                <p>可用的测试患者ID：{', '.join(available_ids)}</p>
                <input type="text" name="patient_id" placeholder="请输入患者ID" required>
                <br><br>
                <button type="submit">登录</button>
            </form>
            <a href="/">← 返回主页</a>
        </body>
        </html>
        """

@app.route('/verify_doctor', methods=['POST'])
def verify_doctor():
    """验证医生身份"""
    verification_code = request.form.get('verification_code')
    if verification_code == '123':
        session['user_type'] = 'doctor'
        return redirect(url_for('doctor_dashboard'))
    else:
        return redirect(url_for('doctor_login'))

@app.route('/verify_patient', methods=['POST'])
def verify_patient():
    """验证患者身份"""
    patient_id = request.form.get('patient_id')
    if patient_id in patient_data:
        session['user_type'] = 'patient'
        session['patient_id'] = patient_id
        return redirect(url_for('patient_chat'))
    else:
        return redirect(url_for('patient_login'))

@app.route('/doctor_dashboard')
def doctor_dashboard():
    """医生控制台"""
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))
    
    try:
        return render_template('doctor_dashboard.html')
    except:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>医生控制台</title>
            <style>
                body { font-family: Arial; margin: 40px; }
                .info-card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }
                .diagnosis-item { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }
                .article-item { border: 1px solid #eee; padding: 15px; margin: 10px 0; background: #fafafa; }
            </style>
        </head>
        <body>
            <h1>👨‍⚕️ 医生控制台</h1>
            <div>
                <input type="text" id="patientId" placeholder="请输入患者ID">
                <button onclick="searchPatient()">查询患者</button>
            </div>
            <div id="results"></div>
            <a href="/logout">退出登录</a>
            
            <script>
            async function searchPatient() {
                const patientId = document.getElementById('patientId').value;
                const response = await fetch('/get_patient_report', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({patient_id: patientId})
                });
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('results').innerHTML = '<div style="color: red;">' + data.error + '</div>';
                    return;
                }
                
                let html = '<div class="info-card">';
                html += '<h3>📋 基本信息</h3>';
                html += '<p><strong>患者ID:</strong> ' + data.patient_info.subject_id + '</p>';
                html += '<p><strong>入院时间:</strong> ' + data.patient_info.admission_time + '</p>';
                html += '<p><strong>出院时间:</strong> ' + data.patient_info.discharge_time + '</p>';
                html += '</div>';
                
                html += '<div class="info-card">';
                html += '<h3>🩺 诊断信息</h3>';
                data.patient_info.diagnosis_codes_translated.forEach(function(diagnosis) {
                    html += '<div class="diagnosis-item">' + diagnosis + '</div>';
                });
                html += '</div>';
                
                if (data.patient_info.procedure_codes_translated.length > 0) {
                    html += '<div class="info-card">';
                    html += '<h3>🏥 治疗操作</h3>';
                    data.patient_info.procedure_codes_translated.forEach(function(procedure) {
                        html += '<div class="diagnosis-item">' + procedure + '</div>';
                    });
                    html += '</div>';
                }
                
                html += '<div class="info-card">';
                html += '<h3>💊 用药记录</h3>';
                html += '<p>' + data.patient_info.medications.slice(0, 10).join(', ') + '</p>';
                html += '</div>';
                
                html += '<div class="info-card">';
                html += '<h3>📚 相关医学文献</h3>';
                data.related_articles.forEach(function(article) {
                    html += '<div class="article-item">';
                    html += '<p><strong>PMID: ' + article.pmid + '</strong></p>';
                    html += '<p>' + article.abstract + '</p>';
                    html += '<a href="' + article.url + '" target="_blank">查看完整文献 →</a>';
                    html += '</div>';
                });
                html += '</div>';
                
                document.getElementById('results').innerHTML = html;
            }
            </script>
        </body>
        </html>
        """

@app.route('/get_patient_report', methods=['POST'])
def get_patient_report():
    """获取患者报告"""
    if session.get('user_type') != 'doctor':
        return jsonify({'error': '未授权访问'})
    
    patient_id = request.json.get('patient_id')
    if patient_id not in patient_data:
        return jsonify({'error': '患者ID不存在'})
    
    patient_info = patient_data[patient_id]
    
    # 获取相关文献
    related_articles = []
    for pmid, article in list(pubmed_articles.items())[:10]:
        related_articles.append({
            'pmid': pmid,
            'abstract': article['abstract'][:200] + '...',
            'url': article['url']
        })
    
    return jsonify({
        'patient_info': patient_info,
        'related_articles': related_articles
    })

@app.route('/patient_chat')
def patient_chat():
    """患者对话页面"""
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))
    
    patient_id = session.get('patient_id')
    patient_info = patient_data.get(patient_id, {})
    
    try:
        return render_template('patient_chat.html', 
                             patient_id=patient_id, 
                             patient_info=patient_info)
    except:
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>AI健康咨询</title></head>
        <body style="font-family: Arial; margin: 40px;">
            <h1>🤖 AI健康助手</h1>
            <p>患者ID: {patient_id}</p>
            <p>您的诊断: {', '.join(patient_info.get('diagnosis_codes_translated', []))}</p>
            
            <div id="messages" style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin: 20px 0;"></div>
            
            <div>
                <input type="text" id="messageInput" placeholder="请输入您的问题" style="width: 70%;">
                <button onclick="sendMessage()">发送</button>
            </div>
            
            <a href="/logout">退出登录</a>
            
            <script>
            async function sendMessage() {{
                const input = document.getElementById('messageInput');
                const message = input.value;
                if (!message) return;
                
                document.getElementById('messages').innerHTML += '<div><strong>您:</strong> ' + message + '</div>';
                input.value = '';
                
                const response = await fetch('/ask_question', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{question: message}})
                }});
                const data = await response.json();
                document.getElementById('messages').innerHTML += '<div><strong>AI:</strong> ' + data.answer + '</div>';
            }}
            </script>
        </body>
        </html>
        """

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """处理患者问题"""
    if session.get('user_type') != 'patient':
        return jsonify({'error': '未授权访问'})
    
    patient_id = session.get('patient_id')
    question = request.json.get('question')
    
    if not question:
        return jsonify({'error': '问题不能为空'})
    # print(f"正在调用API，患者ID: {patient_id}")
    # print(f"问题: {user_question}")
    # 调用AI获取回答
    answer = get_medical_advice(patient_id, question)
    
    return jsonify({
        'answer': answer,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/logout')
def logout():
    """登出"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # 检查templates文件夹
    if not os.path.exists('templates'):
        print("警告：templates文件夹不存在，将使用简单的HTML页面")
        print("请创建templates文件夹并添加模板文件以获得完整的用户界面")
    
    # 启动时加载数据
    if load_patient_data() and extract_pubmed_from_csv():
        print("数据加载成功，可用患者ID：", list(patient_data.keys()))
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("数据加载失败，请检查CSV文件")