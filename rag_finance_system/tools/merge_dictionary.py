"""
merge_dictionary.py
质量过滤 + 合并候选条目到 finance_dictionary.json

用法:
    py -3 rag_finance_system/tools/merge_dictionary.py
    py -3 rag_finance_system/tools/merge_dictionary.py --dry-run  # 预览不写入
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DICT_PATH = PROJECT_ROOT / "data" / "finance_dictionary.json"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "dictionary_candidates.json"


# ── 机构名质量过滤 ──

# 排除 — 泛称政府部门而非具体机构名
EXCLUDE_AUTH_PATTERNS = [
    "国务院有关", "国务院授权", "国务院其他",
    "人民政府", "由国务院", "经国务院", "并报国务院",
    "应当经", "必须经", "应当向", "向国务院",
    "和国务院", "规或者", "办法由", "批准的",
    "方人民政府", "辖市人民政府",
    "当按照国务院", "式或者国务院",
    "报国务院", "报有管理权",
    "由有关", "由交通", "由旅游", "由证券",
    "由保险", "由国务院", "由其",
    "法规和国务院",
    "的主管", "主管部", "主管部门",
    "或者国务院", "或者",
    "国国务院",  # OCR artifact
    # 非金融基础设施管理机构
    "公路管理", "海事管理", "航道管理", "民用航空",
]

# 需要保留的具体机构名 (精确匹配)
KEEP_AUTH_EXACT = {
    "证券交易所": {"short": "证券交易所", "aliases": ["证券交易所", "Stock Exchange"], "category": "证券交易", "level": "中央"},
    "证券登记结算机构": {"short": "证券登记结算机构", "aliases": ["证券登记结算机构", "证券登记结算"], "category": "金融基础设施", "level": "中央"},
    "期货交易所": {"short": "期货交易所", "aliases": ["期货交易所", "Futures Exchange"], "category": "证券交易", "level": "中央"},
    "期货结算机构": {"short": "期货结算机构", "aliases": ["期货结算机构", "期货结算"], "category": "金融基础设施", "level": "中央"},
    "证券监督管理机构": {"short": "证券监管机构", "aliases": ["证券监督管理机构", "证券监管机构"], "category": "证券监管", "level": "中央"},
    "保险监督管理机构": {"short": "保险监管机构", "aliases": ["保险监督管理机构", "保险监管机构"], "category": "保险监管", "level": "中央"},
    "反洗钱行政主管部门": {"short": "反洗钱主管部门", "aliases": ["反洗钱行政主管部门", "反洗钱主管部门"], "category": "合规", "level": "中央"},
    "国务院关税税则委员会": {"short": "关税税则委员会", "aliases": ["国务院关税税则委员会", "关税税则委员会"], "category": "财税审计", "level": "中央"},
    "国家统计局": {"short": "国家统计局", "aliases": ["国家统计局", "统计局", "NBS", "National Bureau of Statistics"], "category": "统计", "level": "中央"},
    "资产评估机构": {"short": "资产评估机构", "aliases": ["资产评估机构", "资产评估"], "category": "金融服务", "level": "中央"},
    "资信评级机构": {"short": "资信评级机构", "aliases": ["资信评级机构", "信用评级机构", "资信评级", "信用评级", "Credit Rating Agency"], "category": "金融服务", "level": "中央"},
    "基金份额登记机构": {"short": "基金份额登记机构", "aliases": ["基金份额登记机构", "基金登记机构", "TA"], "category": "金融基础设施", "level": "中央"},
    "国家畜禽遗传资源委员会": {"short": "畜禽遗传资源委员会", "aliases": ["国家畜禽遗传资源委员会", "畜禽遗传资源委员会"], "category": "农业农村", "level": "中央"},
    "征信机构": {"short": "征信机构", "aliases": ["征信机构", "征信公司", "Credit Reporting Agency"], "category": "征信管理", "level": "中央"},
}


def should_exclude_authority(name):
    """判断机构名是否质量不合格。"""
    # 已经被保留名单覆盖
    if name in KEEP_AUTH_EXACT:
        return False

    # 检查排除模式
    for pattern in EXCLUDE_AUTH_PATTERNS:
        if pattern in name:
            return True

    # 以 "由"/"经"/"应当"/"必须"/"向"/"并"/"和" 开头 → 是短语不是机构名
    if len(name) > 0 and name[0] in ("由", "经", "应", "必", "向", "并", "和", "报", "规", "办", "批", "方", "辖", "当", "式", "国"):
        return True

    # 以 "部" 结尾的泛称 (国务院XX部 虽然也具体但多数是 "国务院XX主管部门")
    if name.endswith("部") and "主管" in name:
        return True

    return False


# ── 术语质量过滤 ──

EXCLUDE_TERMS = {
    "管理机构", "管理部", "监督管理", "管理人员",
    "合法权益", "法律责任", "信息处理", "会计报告",
    "交易者", "许可证",
}

# 需要人工补充定义的术语
TERM_DEFINITIONS = {
    "证券交易": {
        "aliases": ["证券交易", "证券买卖", "Securities Trading"],
        "definition": "依法在证券交易场所进行的证券买卖行为",
        "category": "证券监管",
    },
    "保险人": {
        "aliases": ["保险人", "承保人", "Insurer"],
        "definition": "与投保人订立保险合同，承担赔偿或给付保险金责任的保险公司",
        "category": "保险监管",
    },
    "证券公司": {
        "aliases": ["证券公司", "券商", "Securities Company", "Broker-Dealer"],
        "definition": "经国务院证券监督管理机构批准设立的从事证券经营业务的有限责任公司或股份有限公司",
        "category": "证券监管",
    },
    "公司章程": {
        "aliases": ["公司章程", "章程", "Articles of Association", "AOA"],
        "definition": "公司依法制定的、规定公司基本情况和组织运行规则的文件",
        "category": "公司法",
    },
    "股东会": {
        "aliases": ["股东会", "股东大会", "Shareholders Meeting"],
        "definition": "由全体股东组成的公司最高权力机构",
        "category": "公司法",
    },
    "被保险人": {
        "aliases": ["被保险人", "Insured"],
        "definition": "其财产或人身受保险合同保障，享有保险金请求权的人",
        "category": "保险监管",
    },
    "金融机构": {
        "aliases": ["金融机构", "Financial Institution", "金融组织"],
        "definition": "依法设立的从事金融业务的机构，包括银行、证券公司、保险公司等",
        "category": "金融业务",
    },
    "公司债券": {
        "aliases": ["公司债券", "公司债", "Corporate Bond"],
        "definition": "公司依照法定程序发行、约定在一定期限内还本付息的有价证券",
        "category": "证券监管",
    },
    "税务机关": {
        "aliases": ["税务机关", "税务局", "Tax Authority"],
        "definition": "负责税收征收管理的国家机关",
        "category": "税务",
    },
    "投保人": {
        "aliases": ["投保人", "Policyholder", "Applicant"],
        "definition": "与保险人订立保险合同，并按照保险合同负有支付保险费义务的人",
        "category": "保险监管",
    },
    "发行人": {
        "aliases": ["发行人", "Issuer", "发行主体"],
        "definition": "为筹集资金而发行证券的政府、企业、金融机构等主体",
        "category": "证券监管",
    },
    "抵押权": {
        "aliases": ["抵押权", "抵押", "Mortgage", "Mortgage Right"],
        "definition": "债务人或第三人不转移财产占有而将该财产作为债权的担保",
        "category": "基础法律",
    },
    "上市公司": {
        "aliases": ["上市公司", "Listed Company", "挂牌公司"],
        "definition": "依法在证券交易所上市交易的股份有限公司",
        "category": "公司法",
    },
    "保险公司": {
        "aliases": ["保险公司", "保险机构", "Insurance Company", "Insurer"],
        "definition": "依法设立的经营保险业务的金融机构",
        "category": "保险监管",
    },
    "投资者": {
        "aliases": ["投资者", "投资人", "Investor", "出资人"],
        "definition": "投入资金以获取收益或权益的自然人或机构",
        "category": "证券监管",
    },
    "保险费": {
        "aliases": ["保险费", "保费", "Insurance Premium"],
        "definition": "投保人根据保险合同的约定向保险人支付的费用",
        "category": "保险监管",
    },
    "保险金": {
        "aliases": ["保险金", "保险赔偿", "Insurance Proceeds", "Claim Payment"],
        "definition": "保险事故发生后，保险人向被保险人或受益人支付的金额",
        "category": "保险监管",
    },
    "消费者": {
        "aliases": ["消费者", "Consumer", "顾客"],
        "definition": "为生活消费需要购买、使用商品或接受服务的自然人",
        "category": "消费者保护",
    },
}


def main():
    dry_run = "--dry-run" in sys.argv

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    merged = deepcopy(existing)
    stats = {"laws_added": 0, "authorities_added": 0, "terms_added": 0}
    skipped_laws = []
    skipped_auths = []
    skipped_terms = []

    # ── 合并法规名 ──
    print("=" * 60)
    print("合并法规名...")
    for full_name, info in candidates.get("new_laws", {}).items():
        # 跳过明显非金融的
        non_finance_cats = {"农业农村", "环保", "基础设施", "市场监管", "其他"}
        if info.get("category") in non_finance_cats:
            skipped_laws.append(f"{full_name} (非金融: {info['category']})")
            continue

        entry = {
            "short": info["short"],
            "aliases": info["aliases"],
            "category": info["category"],
            "latest_year": info["latest_year"],
        }
        merged["law_names"][full_name] = entry
        stats["laws_added"] += 1
        print(f"  + {full_name} ({info['category']})")

    # ── 合并机构名 ──
    print("\n合并机构名...")
    # 先加入手动筛选的
    for name, info in KEEP_AUTH_EXACT.items():
        if name in merged["authorities"]:
            continue
        merged["authorities"][name] = info
        stats["authorities_added"] += 1
        print(f"  + {name} ({info['category']})")

    # 再处理候选列表 (过滤后)
    for name, info in candidates.get("new_authorities", {}).items():
        if name in KEEP_AUTH_EXACT:
            continue  # 已加入
        if name in merged["authorities"]:
            continue
        if should_exclude_authority(name):
            skipped_auths.append(name)
            continue
        entry = {
            "short": name,
            "aliases": [name],
            "category": info["category"],
            "level": "中央" if "国务院" in name or "国家" in name else "地方",
        }
        merged["authorities"][name] = entry
        stats["authorities_added"] += 1
        print(f"  + {name} ({info['category']})")

    # ── 合并术语 ──
    print("\n合并术语...")
    for term, info in candidates.get("new_terms", {}).items():
        if term in EXCLUDE_TERMS:
            skipped_terms.append(f"{term} (过于通用)")
            continue

        # 优先使用手动编写的定义
        if term in TERM_DEFINITIONS:
            entry = deepcopy(TERM_DEFINITIONS[term])
        else:
            definition = info.get("definition", "") or info.get("suggested_definition", "")
            if definition == "[需人工补充]":
                definition = ""
            entry = {
                "aliases": info.get("aliases", [term]),
                "category": info.get("category", "其他"),
                "definition": definition,
            }

        merged["terms"][term] = entry
        stats["terms_added"] += 1
        print(f"  + {term} ({entry['category']})")

    # ── 更新元信息 ──
    merged["_version"] = "1.1.0"
    merged["_updated"] = "2026-06-09"

    if dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] 预览结果，未写入文件")
    else:
        with open(DICT_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\n词典已更新: {DICT_PATH}")

    print(f"\n{'='*60}")
    print(f"合并统计:")
    print(f"  新增法规名: {stats['laws_added']} 部")
    print(f"  新增机构名: {stats['authorities_added']} 个")
    print(f"  新增术语:   {stats['terms_added']} 个")
    print(f"  跳过法规名: {len(skipped_laws)} 部 (非金融领域)")
    print(f"  跳过机构名: {len(skipped_auths)} 个 (泛称/短语)")
    print(f"  跳过术语:   {len(skipped_terms)} 个 (过于通用)")
    print(f"\n词典总计:")
    print(f"  术语: {len(merged.get('terms', {}))} 条")
    print(f"  法规名: {len(merged.get('law_names', {}))} 条")
    print(f"  机构: {len(merged.get('authorities', {}))} 个")
    print(f"  缩写: {len(merged.get('abbreviations', {}))} 条")


if __name__ == "__main__":
    main()