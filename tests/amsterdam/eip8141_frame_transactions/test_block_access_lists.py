"""
EIP-8141 frame-transaction interactions with the EIP-7928 block-level
access list: a write discarded by an atomic-batch unroll or a frame
revert must be dropped from the BAL and the slot re-filed as an access.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    BalAccountExpectation,
    BalBalanceChange,
    BalNonceChange,
    BalStorageChange,
    BalStorageSlot,
    Block,
    BlockAccessListExpectation,
    BlockchainTestFiller,
    Bytes,
    Frame,
    FrameReceipt,
    FrameSignature,
    Op,
    Transaction,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

pytestmark = pytest.mark.valid_from("Bogota")

SLOT = 0x01
"""Storage slot the target contracts write."""

WRITTEN_VALUE = 0x42
"""Value the target contracts write to `SLOT`."""

PAYER_POST_BALANCE = 999999999999035855
"""Sponsoring payer's balance after settling the transaction fee."""


@pytest.mark.parametrize(
    "committed",
    [
        pytest.param(True, id="committed"),
        pytest.param(False, id="unrolled"),
    ],
)
def test_bal_atomic_batch_write(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    committed: bool,
) -> None:
    """
    Record an atomic batch's storage write in the BAL only when the
    batch commits; an unrolled batch's write is re-filed as a bare
    access (EIP-8141 atomic batch behavior; EIP-7928 exceptional halts).
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(code=Op.SSTORE(SLOT, WRITTEN_VALUE) + Op.STOP)
    terminator = pre.deploy_contract(
        code=Op.STOP if committed else Op.REVERT(0, 0)
    )

    tx = Transaction(
        sender=sender,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=target,
                gas_limit=200_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=terminator,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                # Succeeds either way; an unrolled batch keeps the status.
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS
                    if committed
                    else Spec.STATUS_FAILURE
                ),
            ],
        ),
    )

    if committed:
        target_expectation = BalAccountExpectation(
            storage_changes=[
                BalStorageSlot(
                    slot=SLOT,
                    slot_changes=[
                        BalStorageChange(
                            block_access_index=1, post_value=WRITTEN_VALUE
                        )
                    ],
                )
            ]
        )
        target_post = Account(storage={SLOT: WRITTEN_VALUE})
    else:
        # Unrolled write: the slot is a bare access, not a change.
        target_expectation = BalAccountExpectation(
            storage_changes=[],
            storage_reads=[SLOT],
        )
        target_post = Account(storage={SLOT: 0})

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={target: target_expectation},
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            sender: Account(nonce=1),
            target: target_post,
        },
    )


def test_bal_atomic_batch_skipped_frame_absent(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Keep a skipped atomic-batch frame's target out of the BAL: the
    first batch frame reverts, the remaining batch frame never executes
    and never accesses its target (EIP-8141 atomic batch behavior).
    """
    sender = pre.fund_eoa()
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    skipped_target = pre.deploy_contract(
        code=Op.SSTORE(SLOT, WRITTEN_VALUE) + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=reverter,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=skipped_target,
                gas_limit=200_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_SKIPPED, gas_used=0),
            ],
        ),
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={skipped_target: None},
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            sender: Account(nonce=1),
            skipped_target: Account(storage={SLOT: 0}),
        },
    )


def test_bal_frame_revert_write_dropped(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Drop a reverting (non-batch) frame's storage write from the BAL,
    re-filing the slot as a bare access (EIP-7928 exceptional halts).
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT, WRITTEN_VALUE) + Op.REVERT(0, 0)
    )

    tx = Transaction(
        sender=sender,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=target,
                gas_limit=200_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                target: BalAccountExpectation(
                    storage_changes=[],
                    storage_reads=[SLOT],
                )
            },
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            sender: Account(nonce=1),
            target: Account(storage={SLOT: 0}),
        },
    )


def test_bal_sponsored_payer_and_sender(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Attribute a sponsored frame transaction in the BAL: a non-sender
    payer's fee balance change and the sender's nonce bump land on
    distinct accounts (EIP-8141 APPROVE_PAYMENT, no sender-equality).
    """
    sender = pre.fund_eoa(amount=10**18)
    payer = pre.fund_eoa(amount=10**18)
    target = pre.deploy_contract(code=Op.SSTORE(SLOT, WRITTEN_VALUE) + Op.STOP)

    tx = Transaction(
        sender=sender,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_PAYMENT,
                target=payer,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=target,
                gas_limit=200_000,
            ),
        ],
        signatures=[
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
            ),
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(payer),
                secret_key=payer.key,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=payer,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                sender: BalAccountExpectation(
                    nonce_changes=[
                        BalNonceChange(block_access_index=1, post_nonce=1)
                    ],
                ),
                payer: BalAccountExpectation(
                    balance_changes=[
                        BalBalanceChange(
                            block_access_index=1,
                            post_balance=PAYER_POST_BALANCE,
                        )
                    ],
                ),
                target: BalAccountExpectation(
                    storage_changes=[
                        BalStorageSlot(
                            slot=SLOT,
                            slot_changes=[
                                BalStorageChange(
                                    block_access_index=1,
                                    post_value=WRITTEN_VALUE,
                                )
                            ],
                        )
                    ],
                ),
            },
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            sender: Account(nonce=1, balance=10**18),
            payer: Account(nonce=0, balance=PAYER_POST_BALANCE),
            target: Account(storage={SLOT: WRITTEN_VALUE}),
        },
    )
