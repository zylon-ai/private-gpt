import asyncio
import json
import random
from datetime import datetime

import aiohttp

CONCURRENCY = 10
# PROJECT_ID = "019f3b6a-9176-730c-8007-897ae3b9c4c4"
# BASE_URL = "https://test1.zylon.ai"
# COOKIE = "default_last_project_=019eaba8-5e6f-78d7-8007-24c5c7a4ff56; ph_phc_hiH2cd6aHtn805k2VKM4Kxc17Hv9EQdATqblxbHsjun_posthog=%7B%22%24device_id%22%3A%2201956278-a10f-75e4-8000-09c2296cabce%22%2C%22distinct_id%22%3A%22019883eb-a7cc-7d26-8000-7788fa98945b%22%2C%22%24sesid%22%3A%5B1783047854043%2C%22019f25ef-1642-706b-ade6-2d317893473c%22%2C1783047853605%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Fdemo.zylon.ai%2F%22%7D%2C%22%24user_state%22%3A%22identified%22%7D; zln_s=9ee31dc4d1b1838a81bb3525d4ad2048; session=Fe26.2**343b8e06efcacd5df6867d956274baf19f583ca57bd299ebbe8e972f0e9e6f9a*jbHMvjN5rLG4vG8AkbBn_w*FZahuF5HaiVHEQfnoKbRb6vk9J0aEeYEysNNccSuk4-i0n7mjXfhELSyN-fG3LP4XOABIkvuqwxI3NN5NYqGIYj5TY0_0kHssWU6cXe_gISaiT_99yqfQRm4E0-6rYZD6H4oEV7Rq7ozy82oENy1FE8fFTjYlw1y5HtRwUDD2vL1B4rTno88YDP7usoCLNdGr-7R0DXWrnJm18W7q8ptZbH8UK15qE_A3lR9ftjOytQfFFO8irdETiex1Tk1kCltLOed8vKZ11FKCYZTBtBi_0GmbkBghVabPwq5spGgBYamZfAZMR9AE3JWw6ifjZxTr9d19zp2zBnoZiBr3owPxU2x55avHsqnrbXg_cDXktprKeo3yXYAkCKVHQEndpI0sZAw1cHYo7njQUMluP0W81AKhvwOGYN3wPD8Fhi78I6WQx5ncUrxSNckvmZWPjNoBrjGlbd5YZHCHMIdD4OiONXiSrMNLbl21oRITPZz43aYiWN3pgKgxdIHU7ux9AreZGcNJw-pVQMvLoDhvWDBU_CM2iCemTY5DElt4TxWFsPoLgnbzoIR4Y10iy6x-anm_i-VIJ462H7e3DJYVewKeF8QaViT806_l3sdTP-5Hc4svdmdBZKlOSDddohEKFbHFN1HgRI8YbWiD7yKeh5JeGbDaCpx7Np_nFw7p-nbyGbZ-aX7Rqbtit-EQ7c3Z1oiP-Kw836X6rlhWafOqhao4q93RMCPCePdy4q0GK7VeyVC3yYVelCrStvMNPh5eX3y8jTmsiQBA9adnraRHOT1_Mt5PbQ63d3YrD6KeoG9-vCh7EBJ79DCQKC0n8CduIjflTjDt4GUKHT84rMItQ4kGx3A-zgL5M1zdVHTRU-ShMNj6x8o76NjwwHSiPh6VTMtpwvl1Q0rUaxpxujX72vxwpB-dphT_bUk7f8QiXyxDoTQ90OefnqjQ4lST0jAf-pQ6GJuLhALh_HMdcyQ67U6sLtSl7nVpy1V2MBehbqHG6r1RzZ4DIbyZNjEKAmH2_OYdNZlcApu9ABWYxj7qVbX0Aeyu9jzQggem8X208I0LGokxusQRC0F-phhyF4Vry47Ioqxj2gj2knfSuIukF2VlMpJuYPCxveM5YVPgpAXWbZEtAf-VrqTZKS8CDMot0u458YmhLCIc7-fMVmOJc3b_IUU8CZ631orp8anjKzB_8U**f232a9d9e81d8bee63680f9c6c30497caea1a49374b156f7ea2a23fa08dd967a*y1UbCehVZaUCSSiE-cQ57n7uHZLd-K-soohDLrtLojk; ph_phc_x4Gsplf9oNLl1M7G4jxVVyZ7Cel2MgIG0TB8m7F5483_posthog=%7B%22%24device_id%22%3A%22019725b9-af01-79ee-8000-87d95db018f5%22%2C%22distinct_id%22%3A%22019c7092-7622-7bb0-8000-5bd5fcb1d4af%22%2C%22%24sesid%22%3A%5B1783408267223%2C%22019f3b6a-50d0-7235-b1ec-8d1e9a64b922%22%2C1783408251087%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Ftest1.zylon.ai%2Fworkspace%2Flogin%3Fredirect%3D%252Fdefault%252Fproject%252F019d1b8a-864a-7da1-8007-e3e1fe2c82d5%252Fchat%252F019dda44-182a-7bce-8009-31855b3888a4%22%7D%2C%22%24user_state%22%3A%22identified%22%7D"
# ARTIFACT_IDS = [
#     "019f3b6b-664e-75be-800b-e5806e51fe23",
#     "019f3b6b-664e-7679-800b-e521898de5ff",
#     "019f3b6b-6653-7e16-800b-ca8e3dd3ec81",
#     "019f3b6b-6662-77d2-800b-c5c618193ca1",
#     "019f3b6b-914a-7071-800b-b3939ac3a232",
#     # "019f3b6b-9234-738c-800b-707e2fafc628",
# ]

PROJECT_ID = "019f3c2f-d5ed-70b5-8007-3ca2f7bef0e8"
BASE_URL = "https://test3.zylon.ai"
COOKIE = "ph_phc_hiH2cd6aHtn805k2VKM4Kxc17Hv9EQdATqblxbHsjun_posthog=%7B%22%24device_id%22%3A%2201956278-a10f-75e4-8000-09c2296cabce%22%2C%22distinct_id%22%3A%22019883eb-a7cc-7d26-8000-7788fa98945b%22%2C%22%24sesid%22%3A%5B1783047854043%2C%22019f25ef-1642-706b-ade6-2d317893473c%22%2C1783047853605%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Fdemo.zylon.ai%2F%22%7D%2C%22%24user_state%22%3A%22identified%22%7D; zln_s=1bd33e3f2a31f5f3090489c9146980e0; ph_phc_x4Gsplf9oNLl1M7G4jxVVyZ7Cel2MgIG0TB8m7F5483_posthog=%7B%22%24device_id%22%3A%22019725b9-af01-79ee-8000-87d95db018f5%22%2C%22distinct_id%22%3A%22019c7092-7622-7bb0-8000-5bd5fcb1d4af%22%2C%22%24sesid%22%3A%5B1783421273177%2C%22019f3c1c-37cc-7e0f-9a7b-f932bacf8cb3%22%2C1783419910091%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Ftest1.zylon.ai%2Fworkspace%2Flogin%3Fredirect%3D%252Fdefault%252Fproject%252F019d1b8a-864a-7da1-8007-e3e1fe2c82d5%252Fchat%252F019dda44-182a-7bce-8009-31855b3888a4%22%7D%2C%22%24user_state%22%3A%22identified%22%7D; default_last_project_=019f3c2f-d5ed-70b5-8007-3ca2f7bef0e8; session=Fe26.2**df7cf409aa5c3554afb5644cf2a9237f6f2db85838f4c16e43b3a0a86f78624f*tEunjaD7cQpqjJ6cNX2bZA*ieoJxw2jqog_GCFM_7b48IIw-aU7BRQY27DwDs9oXqrfrsTq1s-eb0D1R3x2SHSl7otvpxEgYBV383BUOCtj2uo_-fDDRmBn74sej3z1M6L4KlVnv6usFS8w6PVp0it4c2wDanD8Y7H42HRWZFK4P9QVElsnVtX2iz7UGMTgR5j3iF8rz5iux2ved5EsX5YS7o2lF2AdlrEKtd7BuXBMARqbgDkld3gyn6w7QX21rBZSFstoTI9ZMcFLtHb5Nm6d-DgZHRiPhlLKfvI25yiZMOWkkjWuzdyu9KDZDUqQNotSF650rKikjzgTxqFDvDcFHF2_RGQjbd1BFLCZEIuC5jpMumvPFmihKwrCMWWyCZJgFbTzMo5d_3XKn_goMgWbZ3gjqx6quzfceHa4A9m35bROpIT94InBHsThvCqDQ9vdWiSfcZg2_bxB1SeITMoPOr_5MfIu6KzqAD_hcQJ4VkFZO83iFnnQmQvTMCFEQqOnHNlGKWh-Cg6H5CB-lL73RhLFglaUjpPxGR3o6p1qoZ3HznCMRvfeQphrAgtLaCXZ0tLmh2QCy9E3NMtofki3uwBrPtnga5l6NuUM4X9I128qHfG4XEcX_rLTF91ozbcw5ztfwgI2U9N9flxbknlP6MTq3-qb5JRIsf8JxDC7h6M3QWJt0McZVreAKa7g_Dg5nqsGybZKHixRaL8ES9LonYTQKBkceFdmq_Odxr6SPXB5UwRHCttimT9uR46c1fQcI8AghZgNgEQ781WJnARy5VaUeem_A_Hpj6ciNuEV4Tz5QAnbAhxOo9dLe3GdmycfAW434iulG_cOD5ZwfYdkFOcot_yLna2OAY_D-hVeZYkntnQXr0MCnBrwW8qD8HBDGA1PvIuIuL0JXwVZICOgPN34NngdMWCzvDjkc6MrxHkCSUpaMZsn8Eq4M1avJLfa4SUZdj-LbWm0Vf_2KVNyLoGjIkHX7eMQuSHUvqpmxW7x8-gG3n9P6_fDR36TubofiQNl8oDmVIXzIPG_SP-sM-wzP7je-hcoBOrE-Wtn1AxeXS9E4O5HieQ8m_lFqh_eswUc9szSTIEeqpDsUoLL7Oa08RpqmXRhw6ty6pLVO1XbxNaq3cswiXJ082l3vXTZFGblbzk9_s_ZvSRISMVnHe2350yEgv3wsxyNRZp_p0IEp0O4uqitPMK4FkDGJBg**cbf6136440f7caccc1208ef5537fece8a5cf0542d5f744d01c823c8254c0b3b3*Lrmu7VgKL1_hQWnT9V73tI3emAPfeYueSPUYWQrbbzg"
ARTIFACT_IDS = [
    "019f3c30-d7a3-7d75-800b-c0c9980cd573",
    "019f3c36-c0f5-7862-800b-ec8f4629fc3a",
    "019f3c37-2d0a-7613-800b-5bbbf7c4b8f2",
    "019f3c37-2d12-7979-800b-60ce1902922b"
]

TOOLS = [
    {"name": "create_blank_document", "type": "create_blank_document_v1"},
    {"name": "knowledge_search", "type": "semantic_search_v1"},
    {"name": "get_knowledge_base_files", "type": "list_files_v1"},
    {"name": "get_artifact_structure", "type": "get_file_structure_v1"},
    {"name": "get_artifact_content", "type": "get_content_v1"},
    {"name": "create_document", "type": "process_content_v1"},
]

QUESTIONS = [
    "What were Disney's total revenues for fiscal 2022?",
    "How did COVID-19 impact Disney's theme parks?",
    "What is the composition of Disney's DMED segment?",
    "Explain Disney's DTC streaming services.",
    "What were the key restructuring charges in fiscal 2022?",
    "How much did Disney spend on content production?",
    "What is Disney's pension obligation?",
    "Describe Disney's borrowing structure.",
    "What are Disney's major revenue sources?",
    "How did Disney+ subscriber count change?",
    "What is the fair value of Disney's derivatives?",
    "Explain Disney's goodwill impairment testing.",
    "What are Disney's lease obligations?",
    "How does Disney recognize streaming revenue?",
    "What were Disney's capital expenditures?",
    "Describe Disney's equity compensation plans.",
    "What is the status of Shanghai Disney Resort?",
    "How does Disney hedge foreign exchange risk?",
    "What were the operating results for Linear Networks?",
    "Explain Disney's content amortization policy.",
    "What is Disney's effective tax rate?",
    "How much cash from operations?",
    "What are contractual commitments for programming?",
    "Describe Asia Theme Parks ownership.",
    "What intangible assets does Disney hold?",
    "How did Hulu perform in fiscal 2022?",
    "What is commercial paper borrowing capacity?",
    "Explain investment in A+E Television Networks.",
    "What were DPEP results?",
    "How does Disney account for film costs?",
    "What is Disney's dividend policy?",
    "Describe segment operating income breakdown.",
    "What legal matters is Disney involved in?",
    "How does Disney manage interest rate risk?",
    "What are significant accounting policies?",
    "Explain TFCF acquisition integration costs.",
    "What is Disney's credit rating?",
    "How much advertising revenue?",
    "What are multiemployer plan contributions?",
    "Describe VIE consolidation approach.",
    "What was Content License Early Termination impact?",
    "How does Disney value pension assets?",
    "What are unrecognized tax benefits?",
    "Explain depreciation by segment.",
    "What is debt maturity breakdown?",
    "How did ESPN+ perform?",
    "What are deferred tax assets?",
    "Describe merchandise licensing revenues.",
    "What cruise ships are planned?",
    "How does Disney account for licensed content?",
    "What is foreign currency translation approach?",
    "Explain redeemable noncontrolling interests.",
    "What were theatrical distribution results?",
    "How much investment in parks?",
    "What is share repurchase policy?",
    "Describe NBCU relationship regarding Hulu.",
    "What are contingent liabilities?",
    "How classify content individual vs group?",
    "What is Hong Kong Disneyland status?",
    "Explain commodity price risk management.",
    "What were pension calculation assumptions?",
    "How did affiliate fees trend?",
    "What is credit loss allowance approach?",
    "Describe inventory composition.",
    "What equity method investments?",
    "How much interest paid in fiscal 2022?",
    "What are right-of-use assets for leases?",
    "Explain fair value hierarchy.",
    "What were cash flows from discontinued operations?",
    "How recognize theme park admission revenue?",
    "What is weighted average exercise price?",
    "Describe restructuring activities fiscal 2022.",
    "What are bank facility terms?",
    "How did International Channels perform?",
    "What is internal-use software cost policy?",
    "Explain accumulated other comprehensive income components.",
    "What were advertising expenses?",
    "How account for vacation club properties?",
    "What is status of defined contribution plans?",
    "Describe long-lived asset impairment approach.",
    "What were cash and equivalents at year end?",
    "How much contributed to multiemployer plans?",
    "What is accounts receivable aging?",
    "Explain derivative instrument positions.",
    "What were Content Sales/Licensing results?",
    "How handle foreign subsidiary income taxes?",
    "What is accumulated benefit obligation?",
    "Describe DraftKings investment impact.",
    "What are finance lease obligations?",
    "How did DTC losses change?",
    "What is treasury stock position?",
    "Explain supplemental guarantor information.",
    "What were SG&A costs?",
    "How account for participations and residuals?",
    "What is goodwill composition by segment?",
    "Describe cash collateral for derivatives.",
    "What were geographic revenue breakdowns?",
    "How much equity-based compensation?",
    "What is uncertain tax position policy?",
    "Explain segment reporting methodology.",
]

HEADERS = {
    "accept": "application/json, text/plain, */*, text/event-stream",
    "content-type": "application/json",
    "x-org": "default",
    "cookie": COOKIE,
}


async def send_msg(
    session: aiohttp.ClientSession,
    thread_id: str | None,
    msg: str,
    is_first: bool = False,
) -> str | None:
    payload = {
        "thread_id": "new" if is_first else thread_id,
        "artifacts": {
            "document": [
                {"scope": "project", "all_artifacts": False, "artifacts_ids": ARTIFACT_IDS},
            ],
            "connector": [{"scope": "project", "all_artifacts": False, "artifacts_ids": []}],
            "skill": [{"scope": "project", "all_artifacts": False, "artifacts_ids": []}],
        },
        "pgpt": {
            "stream": True,
            "thinking": {"enabled": False},
            "tools": TOOLS,
            "system": {"citations": {"enabled": True}, "use_default_prompt": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": msg}]}],
        },
    }

    if is_first:
        payload["thread_name"] = msg[:50]
        payload["thread_visibility"] = "Private"
        url = f"{BASE_URL}/api/v1/app/project/{PROJECT_ID}/thread/interaction"
    else:
        url = f"{BASE_URL}/api/v1/app/thread/{thread_id}/interaction"

    async with session.post(url, headers=HEADERS, json=payload) as resp:
        resp.raise_for_status()
        async for ctn in resp.content:
            line = ctn.decode("utf-8").strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "interaction":
                        return str(data["interaction"]["thread_id"])
                except:
                    pass
        return None


async def check_status(
    session: aiohttp.ClientSession, thread_id: str | None, idx: int
) -> str | None:
    while True:
        async with session.get(
            f"{BASE_URL}/api/v1/app/thread/{thread_id}/interaction?total_count=true&first=1000&include=User",
            headers=HEADERS,
        ) as resp:
            data = await resp.json()
            if data.get("data") and len(data["data"]) > 0 and idx < len(data["data"]):
                status = data["data"][idx].get("state")
                if status in ["Completed", "Error"]:
                    return str(status)
        await asyncio.sleep(0.5)

    return None


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


async def worker(session: aiohttp.ClientSession, wid: str) -> None:
    try:
        thread_id = None
        log(f"Worker {wid} started")

        questions = QUESTIONS[:]
        random.shuffle(questions)

        for idx, q in enumerate(questions, 1):
            log(f"Worker {wid} [{idx}/{len(questions)}] Sending: {q[:50]}...")
            thread_id = (
                await send_msg(session, thread_id, q, is_first=(idx == 1)) or thread_id
            )
            if not thread_id:
                log(f"Worker {wid} [{idx}/{len(questions)}] ✗ Failed to get thread_id")
                return
            if idx == 1:
                log(f"Worker {wid} thread: {thread_id[:8]}")
            log(f"Worker {wid} [{idx}/{len(questions)}] Waiting for completion...")
            await check_status(session, thread_id, idx - 1)
            log(f"Worker {wid} [{idx}/{len(questions)}] ✓ Completed")

        log(f"Worker {wid} finished all {len(questions)} questions")
    except Exception as e:
        log(f"Worker {wid} encountered an error : {e}")
        raise


async def main() -> None:
    log(f"Starting stress test with {CONCURRENCY} concurrent workers")
    start = datetime.now()

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[worker(session, str(i)) for i in range(CONCURRENCY)])

    duration = (datetime.now() - start).total_seconds()
    total_msgs = CONCURRENCY * len(QUESTIONS)
    log(
        f"Completed {total_msgs} messages in {duration:.1f}s ({total_msgs / duration:.2f} msg/s)"
    )


if __name__ == "__main__":
    asyncio.run(main())
