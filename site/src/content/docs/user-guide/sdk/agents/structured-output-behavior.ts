import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

async function validationExample() {
  // --8<-- [start:validation]
  const today = new Date().toISOString().slice(0, 10)

  const FollowUpSchema = z
    .object({
      action: z.string(),
      dueDate: z.iso.date().refine((value) => value > today, {
        message: `dueDate must be after ${today}`,
      }),
    })
    .describe('A follow-up task for a support ticket')

  const agent = new Agent({ printer: false })
  const result = await agent.invoke(
    'Schedule a follow-up call with the customer next Tuesday.',
    { structuredOutputSchema: FollowUpSchema }
  )
  console.log(result.structuredOutput)
  // --8<-- [end:validation]
}
