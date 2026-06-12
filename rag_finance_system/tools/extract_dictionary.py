"""
extract_dictionary.py
词典自动抽取 — 从 txt_files 扫描法规名变体、机构名、高频术语，生成候选词典条目。

用法:
    py -3 rag_finance_system/tools/extract_dictionary.py

输出:
    data/dictionary_candidates.json  — 候选项，人工审核后合并到 finance_dictionary.json

三个抽取维度:
    1. 法规名 — 从 txt 文件第3行提取官方全名，对比现有词典找出未覆盖的法律
    2. 机构名 — 正则扫描全文，提取监管机构/部委/协会/交易所名
    3. 高频术语 — jieba分词 + TF-IDF，提取高频领域术语
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import jieba.analyse

# ── 配置 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TXT_DIR = Path(__file__).resolve().parent.parent / "src" / "txt_files"
DICT_PATH = PROJECT_ROOT / "data" / "finance_dictionary.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dictionary_candidates.json"


def load_existing_dict():
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# Phase 1: 法规名抽取
# ═══════════════════════════════════════════════════════════════

def extract_law_names_from_files():
    """从每个 txt 文件第3行提取法规全名，返回 {全名: [年份列表]}。"""
    law_versions = defaultdict(list)
    for fpath in sorted(TXT_DIR.glob("*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 跳过空行找到第一个非空行 (通常是第3行)
        law_name = None
        for line in lines[:10]:
            line = line.strip()
            if line and not line.startswith("（") and not line.startswith("目") and "第" not in line[:3]:
                if "中华人民共和国" in line or "全国人民代表大会" in line or len(line) >= 6:
                    law_name = line
                    break

        if not law_name:
            print(f"  [跳过] 无法解析法规名: {fpath.name}")
            continue

        # 从文件名提取年份
        year_match = re.search(r"_(\d{4})\d{4}\.txt$", fpath.name)
        year = year_match.group(1) if year_match else "unknown"

        law_versions[law_name].append(year)

    return dict(law_versions)


def find_new_laws(law_versions, existing_dict):
    """对比现有词典，找出未覆盖的法律。"""
    existing_names = set()
    for full_name in existing_dict.get("law_names", {}):
        existing_names.add(full_name)
        info = existing_dict["law_names"][full_name]
        # 加入短名
        short = info.get("short", "")
        if short:
            existing_names.add(short)
        # 加入所有别名
        for alias in info.get("aliases", []):
            existing_names.add(alias)

    new_laws = {}
    for full_name, years in sorted(law_versions.items()):
        # 检查是否已存在（与全名、短名、别名比对）
        short_name = full_name.replace("中华人民共和国", "")
        is_covered = (
            full_name in existing_names
            or short_name in existing_names
            or full_name in existing_dict.get("law_names", {})
        )
        if is_covered:
            continue

        latest_year = max(years, key=lambda y: int(y) if y.isdigit() else 0)
        new_laws[full_name] = {
            "short": short_name,
            "aliases": [short_name],
            "category": _infer_law_category(full_name),
            "latest_year": latest_year,
            "found_years": years,
        }
    return new_laws


def _infer_law_category(name):
    """根据法规名推断分类。"""
    category_map = [
        (["银行", "商业", "信贷", "存款", "贷款"], "银行监管"),
        (["证券", "基金", "期货", "衍生品"], "证券监管"),
        (["保险"], "保险监管"),
        (["税务", "税收", "企业所得", "个人所得", "增值", "印花", "契税", "关税", "车船", "车辆购置", "资源", "环境", "城市维护", "耕地占用", "船舶吨税", "烟叶"], "税务"),
        (["反洗钱", "反恐怖"], "合规"),
        (["反垄断", "反不正当", "竞争"], "竞争法"),
        (["数据", "个人信息", "网络安全", "信息"], "数据合规"),
        (["公司", "企业破产", "合伙", "个人独资", "市场主体"], "市场主体"),
        (["信托"], "信托监管"),
        (["票据", "支付"], "支付结算"),
        (["知识产权", "专利", "商标", "著作权"], "知识产权"),
        (["合同", "担保", "物权", "侵权"], "基础法律"),
        (["外贸", "外商", "对外", "海商", "出口"], "国际贸易"),
        (["审计"], "审计"),
        (["会计"], "会计"),
        (["资产评估"], "金融服务"),
        (["电子", "电子商务"], "电子商务"),
        (["预算", "国债", "财政"], "财政"),
        (["农业", "农村", "畜牧", "粮食", "农产品", "种子"], "农业农村"),
        (["建筑", "公路", "港口", "航道", "电力", "邮政法"], "基础设施"),
        (["广告", "旅游", "产品质量", "计量", "标准"], "市场监管"),
        (["统计"], "统计"),
        (["海南"], "区域政策"),
        (["民营", "中小"], "市场主体"),
        (["乡村"], "农业农村"),
        (["环保", "防洪", "动物"], "环保"),
    ]
    for keywords, cat in category_map:
        for kw in keywords:
            if kw in name:
                return cat
    return "其他"


# ═══════════════════════════════════════════════════════════════
# Phase 2: 机构名抽取
# ═══════════════════════════════════════════════════════════════

AUTHORITY_PATTERNS = [
    # 国务院XX部/委员会/局/署/办公室
    re.compile(r"国务院[一-鿿]{0,12}(?:部|委员会|局|署|办公室)"),
    # XX监督管理委员会/机构
    re.compile(r"[一-鿿]{2,8}(?:监督管理|监管|管理)(?:委员会|机构)"),
    # XX协会
    re.compile(r"[一-鿿　-〿]{2,12}协会"),
    # XX交易所
    re.compile(r"[一-鿿]{2,8}交易所"),
    # XX银保监局/证监局/监管局/分局
    re.compile(r"[一-鿿]{2,10}(?:银保监局|证监局|保监局|监管局|监管分局)"),
    # 中国人民银行分支机构
    re.compile(r"中国人民银行[一-鿿]{2,10}(?:支行|分行|中心支行)"),
    # 国家XX局
    re.compile(r"国家[一-鿿]{2,10}(?:局|总局|署|委员会)"),
    # 中国XX协会/中心/公司/系统
    re.compile(r"中国[一-鿿]{2,12}(?:协会|中心|系统|公司)"),
    # XX主管部门
    re.compile(r"[一-鿿]{2,8}(?:行政)?主管部门"),
    # XX登记结算机构
    re.compile(r"[一-鿿]{2,10}(?:登记|结算|清算)(?:机构|公司|中心)"),
    # XX征信
    re.compile(r"[一-鿿]{2,8}征信(?:中心|机构|系统|公司)"),
    # XX评价/评级机构
    re.compile(r"[一-鿿]{2,10}(?:评级|评估|评级)(?:机构|公司)"),
    # XX银行/保险/证券行业组织
    re.compile(r"[一-鿿]{2,8}(?:银行|保险|证券|期货|信托|基金)业(?:协会|公会|组织)"),
]

# 排除词 — 过于通用的模式匹配
EXCLUDE_AUTHORITIES = {
    "有关主管部门", "相关主管部门", "行业主管部门", "国务院有关主管部门",
    "国务院有关部门", "有关行政部门", "有关管理部门", "有关监督管理机构",
    "国务院银行业监督管理机构", "国务院保险监督管理机构",
    "国务院证券监督管理机构", "国务院反洗钱行政主管部门",
    "国务院期货监督管理机构", "国务院", "主管部门",
    "相关协会", "行业组织", "自律组织", "有关协会",
}


def extract_authorities_from_text(text):
    """从文本中提取候选机构名。"""
    found = Counter()
    for pattern in AUTHORITY_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group()
            if name not in EXCLUDE_AUTHORITIES and len(name) >= 4:
                found[name] += 1
    return found


def scan_all_for_authorities():
    """扫描所有 txt 文件提取机构名。"""
    all_authorities = Counter()
    for fpath in sorted(TXT_DIR.glob("*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        authorities = extract_authorities_from_text(text)
        all_authorities.update(authorities)
    return all_authorities


def find_new_authorities(found_authorities, existing_dict):
    """找出词典中未覆盖的机构名。"""
    existing_names = set()
    for full_name, info in existing_dict.get("authorities", {}).items():
        existing_names.add(full_name)
        existing_names.add(full_name.lower())
        short = info.get("short", "")
        if short:
            existing_names.add(short.lower())
        for alias in info.get("aliases", []):
            existing_names.add(alias.lower())
        for hist in info.get("historical", []):
            existing_names.add(hist.lower())

    new_auths = {}
    for name, count in found_authorities.most_common(100):
        if name.lower() in existing_names:
            continue
        if count < 3:  # 至少出现3次
            continue
        new_auths[name] = {
            "frequency": count,
            "category": _infer_authority_category(name),
        }
    return new_auths


def _infer_authority_category(name):
    cats = [
        (["银行", "银保监", "金融监管", "金监"], "银行保险监管"),
        (["证监", "证券"], "证券监管"),
        (["保险"], "保险监管"),
        (["人民银行", "央行", "人行"], "中央银行"),
        (["外汇", "外管"], "外汇管理"),
        (["协会", "公会", "自律"], "行业自律"),
        (["交易所"], "证券交易"),
        (["税务", "财政", "审计"], "财税审计"),
        (["发改", "改革"], "宏观调控"),
        (["征信"], "征信管理"),
        (["清算", "结算", "登记"], "金融基础设施"),
        (["评级", "评估"], "金融服务"),
    ]
    for keywords, cat in cats:
        for kw in keywords:
            if kw in name:
                return cat
    return "其他"


# ═══════════════════════════════════════════════════════════════
# Phase 3: 高频术语抽取
# ═══════════════════════════════════════════════════════════════

# 金融领域常见词缀
FINANCE_SUFFIXES = [
    "风险", "资本", "资产", "负债", "权益", "利率", "汇率",
    "管理", "监管", "交易", "投资", "融资", "贷款", "存款",
    "债券", "股票", "基金", "信托", "期货", "期权", "衍生品",
    "保险", "担保", "抵押", "质押", "清算", "结算", "支付",
    "准备金", "拨备", "减值", "损失", "收益", "利润",
    "流动", "杠杆", "敞口", "集中度", "限额", "评级",
    "合规", "内控", "审计", "披露", "报告", "备案",
    "核准", "审批", "许可", "资格", "牌照", "登记",
    "股东", "董事", "监事", "高管", "关联方", "实控",
    "上市", "发行", "承销", "保荐", "做市", "自营",
    "并购", "重组", "破产", "清算", "重整", "和解",
    "反洗钱", "反恐", "制裁", "冻结", "扣划",
    "普惠", "绿色", "小微", "三农", "民营",
    "数字化", "金融科技", "网络安全", "数据", "信息",
    "消费者", "投资者", "存款人", "投保", "投诉",
    "跨域", "跨境", "离岸", "在岸", "自由贸易",
    "宏观审慎", "微观审慎", "系统性", "重要性",
]

FINANCE_PREFIXES = [
    "金融", "银行", "证券", "保险", "信托", "期货",
    "贷款", "存款", "支付", "清算", "结算",
    "信用", "市场", "操作", "流动", "声誉",
    "战略", "合规", "法律", "监管", "税务",
    "会计", "审计", "内控", "公司", "企业",
    "个人", "机构", "境外", "境内", "跨境",
    "资金", "货币", "外汇", "利率", "汇率",
]

# 停用词 — 高频但无领域特异性
STOP_TERMS = {
    "规定", "应当", "不得", "可以", "本法", "法律", "行政法规",
    "有关部门", "以上", "以下", "其他", "或者", "并且",
    "及其", "之日起", "之日起", "包括", "属于", "用于",
    "进行", "提供", "实施", "按照", "根据", "执行",
    "负责", "制定", "建立", "完善", "加强", "促进",
    "发展", "工作", "活动", "经营", "服务", "业务",
    "情况", "条件", "行为", "事项", "内容", "范围",
    "标准", "要求", "办法", "措施", "制度", "机制",
    "部门", "单位", "机构", "组织", "人员", "个人",
    "国家", "政府", "社会", "经济", "市场",
    "一年", "二年", "三年", "五年", "十日", "三十日",
    "一条", "二条", "三条", "四条", "五条",
    "第一款", "第二款", "第三款", "第一项", "第二项",
    "之一", "之二", "之一",
    "限期", "改正", "责令", "罚款", "处分",
    "申请", "批准", "同意", "备案", "报告",
    "人民", "举报", "检举", "控告", "投诉",
}


def extract_terms_with_tfidf():
    """用 jieba TF-IDF 从全部 txt 中提取高频关键词。"""
    # 合并所有文本
    all_text = ""
    for fpath in sorted(TXT_DIR.glob("*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"

    # 添加金融领域自定义词典
    for suffix in FINANCE_SUFFIXES:
        jieba.add_word(suffix, freq=100)
    for prefix in FINANCE_PREFIXES:
        jieba.add_word(prefix, freq=100)

    # 使用 TF-IDF 提取关键词
    keywords = jieba.analyse.extract_tags(
        all_text,
        topK=500,
        withWeight=True,
        allowPOS=("n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an"),
    )

    # 过滤
    filtered = []
    for word, weight in keywords:
        if len(word) < 2:
            continue
        if word in STOP_TERMS:
            continue
        if word.isdigit():
            continue
        # 必须是中文为主
        chinese_chars = sum(1 for c in word if "一" <= c <= "鿿")
        if chinese_chars < 2:
            continue
        # 过滤掉数字+中文单位组合
        if re.match(r"^[\d一二三四五六七八九十百千万亿]+", word):
            continue
        filtered.append((word, round(weight, 4)))

    return filtered


def _suggest_term_definition(term, existing_dict):
    """为候选术语推荐定义 — 基于词缀推断。"""
    suffix_defs = {
        "风险": "可能造成损失的不确定性因素",
        "资本": "企业所有者投入的资金或监管要求的资本",
        "资产": "企业拥有或控制的经济资源",
        "负债": "企业承担的现时义务",
        "权益": "企业资产扣除负债后的剩余利益",
        "利率": "一定时期内利息额与借贷资金额的比率",
        "管理": "为实现目标而进行的规划、组织、领导和控制活动",
        "监管": "监管机构依法对市场主体实施的监督和管理",
        "交易": "买卖双方进行资产交换的行为",
        "贷款": "金融机构向借款人提供的、约定还本付息的资金",
        "债券": "发行人向投资者发行的债务凭证",
        "基金": "为特定目的设立、由专业管理人管理的资金池",
        "保险": "投保人支付保费、保险人承担风险保障的制度安排",
        "担保": "为保障债权实现而提供的信用增信措施",
        "清算": "终止法人资格时清理债权债务的程序",
        "结算": "交易双方完成资金与资产交割的过程",
        "支付": "付款人向收款人转移货币资金的行为",
        "拨备": "为预期损失提前计提的准备金",
        "合规": "企业经营活动符合法律法规和监管要求的状态",
        "披露": "按规定向公众或监管机构公开信息的行为",
    }
    for suffix, definition in suffix_defs.items():
        if term.endswith(suffix) and term != suffix:
            return definition
    return ""


def find_new_terms(tfidf_terms, existing_dict):
    """找出词典中未覆盖的高频术语。"""
    existing_term_names = set()
    for canonical in existing_dict.get("terms", {}):
        existing_term_names.add(canonical.lower())
        for alias in existing_dict["terms"][canonical].get("aliases", []):
            existing_term_names.add(alias.lower())

    new_terms = {}
    for word, weight in tfidf_terms:
        if word.lower() in existing_term_names:
            continue
        if len(word) < 3:  # 单字/双字太泛
            continue

        # 要求有金融领域特征
        has_finance_feature = (
            any(suffix in word for suffix in FINANCE_SUFFIXES)
            or any(word.startswith(prefix) for prefix in FINANCE_PREFIXES)
        )
        if not has_finance_feature:
            continue

        definition = _suggest_term_definition(word, existing_dict)
        new_terms[word] = {
            "tfidf_weight": weight,
            "aliases": [word],
            "definition": definition,
            "category": _infer_term_category(word),
            "suggested_definition": definition or "[需人工补充]",
        }

    return new_terms


def _infer_term_category(term):
    cats = [
        (["银行", "信贷", "存款", "贷款", "拨备", "资本充足", "不良", "授信"], "银行监管"),
        (["证券", "股票", "债券", "基金", "IPO", "发行", "承销"], "证券监管"),
        (["保险", "偿付", "精算", "再保"], "保险监管"),
        (["期货", "衍生品", "期权"], "期货监管"),
        (["信托", "受益权"], "信托监管"),
        (["票据", "支付", "结算", "清算"], "支付结算"),
        (["税", "所得", "增值", "印花", "契税"], "税务"),
        (["公司", "股东", "董事", "监事", "章程", "注册资本", "破产"], "公司法"),
        (["反洗钱", "反恐", "AML", "KYC", "CDD"], "合规"),
        (["反垄断", "竞争", "经营者集中"], "竞争法"),
        (["数据", "个人信息", "隐私", "网络"], "数据合规"),
        (["利率", "汇率", "货币", "宏观审慎", "准备金", "MLF", "LPR"], "货币政策"),
        (["外汇", "跨境", "资本项目", "QDII", "QFII"], "外汇管理"),
        (["消费者", "投资者", "适当性"], "消费者保护"),
        (["征信", "信用"], "征信管理"),
        (["知识", "专利", "商标", "著作权", "版权"], "知识产权"),
    ]
    for keywords, cat in cats:
        for kw in keywords:
            if kw.lower() in term.lower():
                return cat
    return "其他"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("词典自动抽取 — 从 txt_files 扫描候选条目")
    print("=" * 60)

    existing = load_existing_dict()
    print(f"\n现有词典: {existing['_version']}, 更新于 {existing['_updated']}")
    print(f"  术语: {len(existing.get('terms', {}))} 条")
    print(f"  法规名: {len(existing.get('law_names', {}))} 条")
    print(f"  机构: {len(existing.get('authorities', {}))} 条")
    print(f"  缩写: {len(existing.get('abbreviations', {}))} 条")

    # ── Phase 1: 法规名 ──
    print("\n" + "-" * 40)
    print("Phase 1: 从 txt 文件提取法规全名...")
    law_versions = extract_law_names_from_files()
    print(f"  从 {len(law_versions)} 部独特性法规中扫描")
    new_laws = find_new_laws(law_versions, existing)
    print(f"  新发现法规: {len(new_laws)} 部")
    for name in new_laws:
        print(f"    + {name} ({new_laws[name]['latest_year']})")

    # ── Phase 2: 机构名 ──
    print("\n" + "-" * 40)
    print("Phase 2: 正则扫描机构名...")
    found_authorities = scan_all_for_authorities()
    print(f"  共扫描到 {len(found_authorities)} 个候选机构名 (含重复)")
    new_auths = find_new_authorities(found_authorities, existing)
    print(f"  新发现机构: {len(new_auths)} 个")
    for name, info in sorted(new_auths.items(), key=lambda x: -x[1]["frequency"])[:20]:
        print(f"    + {name} (出现{info['frequency']}次, {info['category']})")

    # ── Phase 3: 高频术语 ──
    print("\n" + "-" * 40)
    print("Phase 3: jieba TF-IDF 提取高频术语...")
    tfidf_terms = extract_terms_with_tfidf()
    print(f"  TF-IDF 提取 {len(tfidf_terms)} 个关键词")
    new_terms = find_new_terms(tfidf_terms, existing)
    print(f"  新发现术语: {len(new_terms)} 个")
    for term, info in list(new_terms.items())[:30]:
        print(f"    + {term} (权重{info['tfidf_weight']}, {info['category']})")

    # ── 汇总输出 ──
    candidates = {
        "_description": "词典自动抽取候选项 — 人工审核后合并到 finance_dictionary.json",
        "_generated": "2026-06-09",
        "_summary": {
            "new_laws": len(new_laws),
            "new_authorities": len(new_auths),
            "new_terms": len(new_terms),
        },
        "new_laws": new_laws,
        "new_authorities": {
            name: info for name, info in
            sorted(new_auths.items(), key=lambda x: -x[1]["frequency"])
        },
        "new_terms": {
            term: info for term, info in
            sorted(new_terms.items(), key=lambda x: -x[1]["tfidf_weight"])
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"候选条目已写入: {OUTPUT_PATH}")
    print(f"  新法规名: {len(new_laws)} 部")
    print(f"  新机构名: {len(new_auths)} 个")
    print(f"  新术语:   {len(new_terms)} 个")
    print(f"\n请人工审核候选文件，确认后合并到词典。")


if __name__ == "__main__":
    main()